import logging
import random
import os
import torch
import torch.nn as nn
import numpy as np

import torchvision
import open_clip
import cn_clip.clip as cn_clip
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForSequenceClassification

import adapters
from sentence_transformers import SentenceTransformer

from training.distributed import is_master
from training.projection import DINOHead
import training.transforms

from loss import NEED_LOGIT_SCALE, NEED_PROTOTYPE_LAYER
from contextlib import suppress

AVALIABLE_TEXT_MODEL_BUILDER = ['openclip', 'chineseclip', 'huggingface', 'sbert']
AVALIABLE_IMAGE_MODEL_BUILDER = ['openclip', 'chineseclip', 'torchvision', "torchhub"]

from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForSequenceClassification
from transformers import BertConfig
import torch.nn.functional as F
from model.xbert import BertModel

def get_model(args):
    logging.info(f'Builing model for rank {args.rank}')
    
    # === text model === #
    if is_master(args):
        logging.info(f'Loading [{args.text_model}] as text model via [{args.text_model_builder}]. Pretrained={args.pretrained_text_model}')
    
    if args.text_model_builder=='openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.text_model,
            pretrained=args.text_model_tag if args.pretrained_text_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            cache_dir=os.path.join(args.cache_dir, 'open_clip')
        )
        if args.load_clip_weights:
            checkpoint = torch.load(args.load_clip_weights_from, map_location="cpu")
            msg = CLIP_model.load_state_dict(checkpoint)
            logging.info(f"text: {msg}")
        CLIP_model.visual = None
        text_backbone = CLIP_model

        tokenizer = open_clip.tokenize
        args.text_width, args.text_dim = text_backbone.text_projection.size()
        text_backbone.layers = open_clip.get_model_config(args.text_model)['text_cfg']['layers']
                    
        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
            
    elif args.text_model_builder=='chineseclip':
        CLIP_model, preprocess_val = cn_clip.load_from_name(
            name=args.text_model, 
            device=args.device, 
            download_root=os.path.join(args.cache_dir, 'cn_clip')
            )
        preprocess_train = preprocess_val # TODO: add data augmentations
        
        CLIP_model.visual = None
        text_backbone = CLIP_model
        tokenizer = cn_clip.tokenize
        args.text_width, args.text_dim = text_backbone.text_projection.size()
                    
        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
        
    elif args.text_model_builder=='huggingface':
        if not args.pretrained_text_model and is_master(args):
            logging.info(f'huggingface-transormer uses pretrained weight by default!')
        config = AutoConfig.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        
        if 'Taiyi' in args.text_model:
            # Taiyi models need classification heads, using AutoModel will skip loading them.
            # See https://huggingface.co/IDEA-CCNL/Taiyi-Roberta-124M-D-v2
            text_backbone = BertForSequenceClassification.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        else:
            text_backbone = AutoModel.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        args.text_dim = config.hidden_size
        args.text_width = None
                
        if args.adapter is not None:
            if args.adapter=='bottleneck_adapter':
                config = adapters.AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
            elif args.adapter=='prefix_tuning':
                config = adapters.PrefixTuningConfig(flat=False, prefix_length=30)
            elif args.adapter=='lang_adapter':
                config = adapters.SeqBnInvConfig()
            elif args.adapter=='lora_adapter':
                config = adapters.LoRAConfig(r=8, alpha=16)
            elif args.adapter=='dummy':
                config = adapters.CompacterConfig()
            elif args.adapter=='ia3_adapter':
                config = adapters.IA3Config()
            elif args.adapter=='mam_adapter':
                config = adapters.MAMConfig()
            elif args.adapter=='unipelt':
                config = adapters.UniPELTConfig()
            else:
                raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
            
            logging.info(f'[Adapter]: Using adapter: {args.adapter}.')
            text_backbone.add_adapter(args.adapter, config=config)
            text_backbone.train_adapter(args.adapter)
            
    elif args.text_model_builder=='sbert':
        if not args.pretrained_text_model and is_master(args):
            logging.info(f'Sentence-transormer uses pretrained weight by default!')
        text_backbone = SentenceTransformer(args.text_model, device=args.device).to(args.device)
        tokenizer = None
        args.text_dim = text_backbone.get_sentence_embedding_dimension()
        args.text_width = args.text_dim
        
        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
    
    else:
        raise RuntimeError(f'text model builder "{args.text_model_builder}" is not supported.')
    
    
    # === image model === #
    if is_master(args):
        logging.info(f'Loading [{args.image_model}] as image model via [{args.image_model_builder}]. Pretrained={args.pretrained_image_model}')
    
    if args.image_model_builder=='openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.image_model,
            pretrained=args.image_model_tag if args.pretrained_image_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            cache_dir=os.path.join(args.cache_dir, 'open_clip')
        )
        if args.load_clip_weights:
            checkpoint = torch.load(args.load_clip_weights_from, map_location="cpu")
            msg = CLIP_model.load_state_dict(checkpoint)
            logging.info(f"image: {msg}")

        image_backbone = CLIP_model.visual
        args.image_dim = image_backbone.output_dim
        image_backbone.layers = open_clip.get_model_config(args.image_model)['vision_cfg']['layers']
        if type(image_backbone.layers) == list:
            image_backbone.layers = len(image_backbone.layers)
        if 'RN' in args.image_model:
            image_backbone.arch = 'ResNet'
            image_backbone.layers += 2 # stem and attention pooling accont for two layers
        elif 'ViT' in args.image_model:
            image_backbone.arch = 'ViT'
        else:
            raise RuntimeError(f'Unrecognized image backbone architechture')

        
    elif args.image_model_builder=='chineseclip':
        CLIP_model, preprocess_val = cn_clip.load_from_name(
            name=args.text_model, 
            device=args.device, 
            download_root=os.path.join(args.cache_dir, 'cn_clip')
            )

        preprocess_train = preprocess_val # TODO: add data augmentations        
        image_backbone = CLIP_model.visual
        args.image_dim = image_backbone.output_dim
    
    elif args.image_model_builder=='torchvision':
        os.environ['TORCH_HOME'] = os.path.join(args.cache_dir, 'torchvision')
        image_backbone = torchvision.models.__dict__[args.image_model](pretrained=args.pretrained_image_model, num_classes=1000)

        LAST_FC = ['resnet', 'shufflenet', 'convnext', 'regnet', 'inception']
        CLASSIFIER_WITH_DROPOUT = ['alexnet', 'squeezenet', 'mnasnet', 'efficientnet']
        CLASSIFIER_WITHOUT_DROPOUT = ['mobilenet', 'vgg', 'densenet', 'googlenet']
        LAST_HEAD = ['vit']

        if 'resnet' in args.image_model or 'shufflenet' in args.image_model or 'convnext' in args.image_model or 'regnet' in args.image_model or 'inception' in args.image_model:
            image_backbone.output_dim = image_backbone.fc.weight.shape[1]
            image_backbone.fc=torch.nn.Identity()
        if 'alexnet' in args.image_model or 'squeezenet' in args.image_model or 'mnasnet' in args.image_model or 'efficientnet' in args.image_model: 
            # there are dropout layer between first linear layer in the classifier # TODO: squeeze net has conv2d instead of linear
            image_backbone.output_dim = image_backbone.classifier[1].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        if 'mobilenet' in args.image_model or 'vgg' in args.image_model or 'densenet' in args.image_model or 'googlenet' in args.image_model :
            image_backbone.output_dim = image_backbone.classifier[0].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        if 'vit' in args.image_model:
            image_backbone.output_dim = image_backbone.hidden_dim
            image_backbone.heads=torch.nn.Identity()
            image_backbone.head=torch.nn.Identity()
        image_backbone.to(device=args.device)

        preprocess_train = training.transforms.get_preprocess(image_resolution=args.image_resolution, is_train=True, aug=args.augmentation)
        preprocess_val = training.transforms.get_preprocess(image_resolution=args.image_resolution, is_train=False, aug=None)
    
    elif args.image_model_builder=='torchhub':
        if not args.pretrained_image_model and is_master(args):
            logging.info(f'Torch hub uses pretrained weight by default!')

        torch.hub.set_dir(os.path.join(args.cache_dir, 'torchhub'))
        
        # https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True 

        image_backbone = torch.hub.load(args.image_model_tag, args.image_model)
        if 'resnet' in args.image_model:
            if 'vicreg' in args.image_model_tag:
                image_backbone.output_dim = 2048 # TODO: check ResNet-50 (x2) and ResNet-200 (x2)
            
        if 'vit' in args.image_model or 'xcit' in args.image_model:            
            image_backbone.output_dim = image_backbone.norm.weight.size(0)

        if 'regnety' in args.image_model:            
            image_backbone.output_dim = 2048

        image_backbone.to(device=args.device)
        preprocess_train, preprocess_val = training.transforms.preprocess_train, training.transforms.preprocess_val

    else:
        raise RuntimeError(f'image model builder "{args.image_model_builder}" is not supported.')
   
    # Set 'param.required_grad' to implement partial finetune
    for name, param in text_backbone.named_parameters():
        param.requires_grad = False if args.lock_text_model else True
        if args.lock_text_partial != '':
            for keyword in args.lock_text_partial.split(','):
                if keyword.replace('!', '') in name:
                    if '!' in keyword:
                        param.requires_grad = True
                        if args.lock_text_model:
                            break
                    else:
                        param.requires_grad = False
                        if not args.lock_text_model:
                            break
                    
    for name, param in image_backbone.named_parameters():
        param.requires_grad = False if args.lock_image_model else True
        if args.lock_image_partial != '':
            for keyword in args.lock_image_partial.split(','):
                if keyword.replace('!', '') in name:
                    if '!' in keyword:
                        param.requires_grad = True
                        if args.lock_image_model:
                            break
                    else:
                        param.requires_grad = False
                        if not args.lock_image_model:
                            break

    model = ItraModel(
        text_backbone=text_backbone, 
        image_backbone=image_backbone, 
        tokenizer=tokenizer, 
        args=args
        )
        
    return model, preprocess_train, preprocess_val, preprocess_val



class ItraModel(nn.Module):
    def __init__(self, text_backbone, image_backbone, tokenizer, args, temperature=0.07, margin=0.2) -> None:
        super().__init__()
        self.device = args.device
        self.text_model = args.text_model
    
    # text backbone
        self.text_backbone = text_backbone
        self.text_pooler = args.text_pooler
        if self.text_pooler!= 'cls':
            self.text_backbone.pooler = nn.Identity()
        self.text_dim = args.text_dim
        self.text_width = args.text_dim
        self.tokenizer = tokenizer        
        self.text_model_builder = args.text_model_builder
        self.image_model_builder = args.image_model_builder
        self.max_seq_length = args.max_seq_length
            
        self.image_context = torch.no_grad if (
            args.lock_image_model and 
            '!' not in args.lock_image_partial
            ) else suppress 
            
        self.text_context = torch.no_grad if (
            args.lock_text_model and 
            '!' not in args.lock_text_partial and 
            args.adapter is None and
            not args.prompt
            ) else suppress
        
        if is_master(args):
            logging.info(f'Calculate gradients for image backbone?\t{self.image_context==suppress}')
            logging.info(f'Calculate gradients for text backbone?\t{self.text_context==suppress}')
        
        # TODO: CoOp text prompt
        if args.prompt:
            assert args.text_model_builder=='openclip' # CoOp style prompt only supports OpenCLIP models
            self.prompt = nn.Parameter(torch.empty(args.n_prompt, args.text_width))
            torch.nn.init.normal_(self.prompt, std=0.02)
            self.n_prompt = args.n_prompt
        else:
            self.prompt = None

    # image backbone
        self.image_backbone = image_backbone
        self.image_dim = image_backbone.output_dim
        self.image_model_tag = args.image_model_tag

    
    # text projection head
        if args.text_head_n_layers > 0 or args.loss in NEED_PROTOTYPE_LAYER:
            if args.image_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.image_dim # adaption layer
            self.text_projection_head = DINOHead(
                in_dim=self.text_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.text_head_n_layers, skip_last_layer=args.loss not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            
            # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.loss in NEED_PROTOTYPE_LAYER and args.teacher=='text':
                for param in self.text_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.text_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Text backbone do not append projection head, so set args.joint_projection_dim = self.text_dim')
            args.joint_projection_dim = self.text_dim

    # image projection head
        if args.image_head_n_layers > 0 or args.loss in NEED_PROTOTYPE_LAYER:
            if args.text_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.text_dim # adaption layer
            self.image_projection_head = DINOHead(
                in_dim=self.image_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.image_head_n_layers, skip_last_layer=args.loss not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            # FIXME? # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.loss in NEED_PROTOTYPE_LAYER and args.teacher=='image':
                for param in self.image_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.image_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Image backbone do not append projection head so set args.joint_projection_dim = self.image_dim')
            args.joint_projection_dim = self.image_dim

        if args.loss in NEED_LOGIT_SCALE:
            if hasattr(self.text_backbone, 'logit_scale'):
                self.logit_scale = self.text_backbone.logit_scale 
                self.text_backbone.logit_scale = None
            else:
                self.logit_scale = torch.autograd.Variable(torch.ones(1) * np.log(1 / args.logit_scale)).to(self.device)
            self.logit_scale = nn.Parameter(self.logit_scale)
            self.logit_scale.requires_grad = True
        else:
            self.logit_scale = torch.zeros(1)
        self.to(self.device)

        # 融合编码器
        dim_str = 512
        self.proj_image = nn.Linear(768, dim_str)
        self.proj_text = nn.Linear(512, dim_str)
        self.fusion_model = FusionModel(pretrained_path=None)
        self.fc_image = nn.Linear(dim_str, dim_str)
        self.fc_text = nn.Linear(dim_str, dim_str)
        for param in self.fusion_model.parameters():
            param.requires_grad = True
        for name, param in self.fusion_model.named_parameters():
            if name in ['fusion_model.pooler.dense.bias', 
                        'fusion_model.pooler.dense.weight', 
                        'fusion_model.embeddings.word_embeddings.weight']:
                param.requires_grad = False
        for param in self.proj_image.parameters():
            param.requires_grad = True
        for param in self.proj_text.parameters():
            param.requires_grad = True
        for param in self.fc_image.parameters():
            param.requires_grad = True
        for param in self.fc_text.parameters():
            param.requires_grad = True  

        it_encoder_layer = nn.TransformerEncoderLayer(dim_str, 8, 1024, 0.1, batch_first=True)
        self.it_encoder = nn.TransformerEncoder(it_encoder_layer, 2)

        # self.rec_decoder = FusionModel(pretrained_path=None)

        # loss
        self.rec_loss = nn.MSELoss()
        self.info_nce_loss = BiDirectionalInfoNCELoss(temperature=temperature)
        self.triplet_loss = BiDirectionalTripletLoss(margin=margin)

        self.loss = 0 
        self.to(self.device)
        self.text_hidden = torch.tensor(1)

    def reinit_logit_scale(self, logit_scale):
        self.logit_scale = torch.nn.parameter.Parameter(torch.ones(1) * np.log(1 / logit_scale))#.to(self.device)
        #self.logit_scale.to(self.device)
        self.to(self.device)

    def encode_image(self, images, projection=False):
        with self.image_context():
            if self.image_model_builder=='chineseclip': 
                images = images.type(self.image_backbone.conv1.weight.dtype)
            image_features = self.image_backbone(images)
            if 'vicregl' in self.image_model_tag:
                image_features = image_features[1]
        if projection:
            image_features = self.image_projection_head(image_features)
        return image_features.float()

    # sentence-transformers API
    def encode(self, sentences, batch_size=32, show_progress_bar=None, convert_to_numpy=True, convert_to_tensor=True, use_pooler=False):
        with torch.no_grad():
            def _text_length(text):
                if isinstance(text, dict):              #{key: value} case
                    return len(next(iter(text.values())))
                elif not hasattr(text, '__len__'):      #Object has no len() method
                    return 1
                elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
                    return len(text)
                else:
                    return sum([len(t) for t in text])      #Sum of length of individual strings

            all_embeddings = []
            length_sorted_idx = np.argsort([_text_length(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in range(0, len(sentences), batch_size):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                embeddings = self.encode_text(sentences_batch, projection=True, use_pooler=use_pooler).cpu()
                all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                all_embeddings = torch.stack(all_embeddings)
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings
    
    def encode_text(self, texts, projection=False, use_pooler=True):
        with self.text_context():
            if self.text_model_builder in ['openclip']:
                # TODO: support CoOp-style prompting (CoOp for retrieval finetuning?)
                context_length = (77 - self.n_prompt) if self.prompt is not None else 77
                texts = self.tokenizer(texts, context_length=context_length).to(self.device)
                def open_clip_forward(texts):
                    x = self.text_backbone.token_embedding(texts)  # [batch_size, n_ctx, d_model] (bs, 77-args.n_prompts, 512)
                    if self.prompt is not None:
                        batch_prompt = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1)
                        x = torch.cat([x[:, :1, :], batch_prompt, x[:, 1:, :]], dim=1)
                    x = x + self.text_backbone.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.text_backbone.ln_final(x) # [batch_size, n_ctx, transformer.width]
                    self.text_hidden = x
                    # take features from the eot embedding (eot_token is the highest number in each sequence)
                    x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
                    return x
                text_features = open_clip_forward(texts)
            
            elif self.text_model_builder=='chineseclip': 
                # text_features = self.text_backbone.encode_text(self.tokenizer(texts).to(self.device))
                # Re-write encode_text() function to avoid reading missing .dtype()
                # For https://github.com/OFA-Sys/Chinese-CLIP/blob/ce534cc8c0dde1206cd3e3ddf7e4023455c83450/cn_clip/clip/model.py#L375
                
                def chineseclip_encode_text(text):
                    pad_index = self.text_backbone.tokenizer.vocab['[PAD]']
                    attn_mask = text.ne(pad_index).type(self.text_backbone.text_projection.dtype)
                    x = self.text_backbone.bert(text, attention_mask=attn_mask)[0].type(self.text_backbone.text_projection.dtype) # [batch_size, seq_length, hidden_size]
                    return x[:, 0, :] @ self.text_backbone.text_projection

                text_features = chineseclip_encode_text(self.tokenizer(texts).to(self.device)).float()

            elif self.text_model_builder=='sbert':            
                texts = self.text_backbone.tokenize(texts)
                texts = {
                    'input_ids': texts['input_ids'].to(self.device),
                    'attention_mask': texts['attention_mask'].to(self.device)
                    }
                text_features = self.text_backbone(texts)
                sentence_embedding = text_features['sentence_embedding']
                text_features = sentence_embedding
                #token_embeddings = text_features['token_embeddings']
                #text_features = token_embeddings[:, 0, :].contiguous()

            elif self.text_model_builder=='huggingface':           
                # Preprocess
                if self.text_pooler == 'PromptBERT':
                    texts_lengths = [] # memorize the number of token of each sentence for position id padding
                    for t in range(len(texts)):
                        encoded_sentence = self.tokenizer.encode(texts[t], truncation=True, max_length=self.max_seq_length)
                        texts_lengths.append(len(encoded_sentence))
                        sentence = self.tokenizer.decode(encoded_sentence, skip_special_tokens=True)

                        if self.text_model in ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased']:
                            if random.random() > 0.5 or not self.training:
                                texts[t] = f'The sentence of "{sentence}" means {self.tokenizer.mask_token}.'
                            else:
                                texts[t] = f'This sentence : "{sentence}" means {self.tokenizer.mask_token}.'
                        else: # roberta
                            if random.random() > 0.5 or not self.training:
                                texts[t] = f"This sentence : '{sentence}' means {self.tokenizer.mask_token}."
                            else:
                                texts[t] = f"The sentence : '{sentence}' means {self.tokenizer.mask_token}."

                    texts_lengths = np.array(texts_lengths)
                    encoded_input = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")

                else:
                    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_length)

                # To GPU
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].to(self.device),
                    'attention_mask': encoded_input['attention_mask'].to(self.device)
                    }

                # Forward
                outputs = self.text_backbone(**encoded_input, output_hidden_states=True, return_dict=True)
                    # last_hidden = outputs.last_hidden_state   # (batch_size, sequence_length, hidden_size)
                    # pooler_output = outputs.pooler_output     # (batch_size, hidden_size)
                    # hidden_states = outputs.hidden_states     # (batch_size, sequence_length, hidden_size) x layers (tuple)

                # Pooling
                # use_pooler = True if self.training else False # TODO: there is a conflict between retrieval evaluation and sts evaluation...
                if self.text_pooler=='mean':
                    text_features = mean_pooling(outputs.last_hidden_state, encoded_input['attention_mask'])

                elif self.text_pooler=='cls' and use_pooler:
                    text_features = outputs.pooler_output
                
                elif (self.text_pooler=='cls' and not use_pooler) or (self.text_pooler == 'cls_before_pooler'):
                    text_features = outputs.last_hidden_state[:, 0].contiguous()

                elif self.text_pooler == 'PromptBERT':
                    # Retrieve [mask] token
                    text_features = outputs.last_hidden_state[encoded_input['input_ids'] == self.tokenizer.mask_token_id].contiguous()
                    # Template Denoising
                    with torch.no_grad():
                        if self.text_model in ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased']:
                            encoded_input_delta = self.tokenizer(f'The sentence of " " means {self.tokenizer.mask_token}.', return_tensors="pt")
                        else:
                            encoded_input_delta = self.tokenizer(f"This sentence : ' ' means {self.tokenizer.mask_token}.", return_tensors="pt")
                            
                        encoded_input_delta = {
                            'input_ids': encoded_input_delta['input_ids'].repeat(len(texts), 1).to(self.device),
                            'attention_mask': encoded_input_delta['attention_mask'].repeat(len(texts), 1).to(self.device),
                        }
                        delta_position_ids = torch.arange(len(encoded_input_delta['input_ids'][0])).long().repeat(len(texts), 1)
                        # (0) <Start> | (1) This | (2) sentence | (3) of/: | (4) "/' | (5) {sentence} ...
                        delta_position_ids[:,5:] += texts_lengths.reshape(len(texts), 1) 
                        delta = self.text_backbone(**encoded_input_delta, position_ids=delta_position_ids.to(self.device), output_hidden_states=True, return_dict=True)
                        delta = delta.last_hidden_state[encoded_input_delta['input_ids'] == self.tokenizer.mask_token_id]
                    text_features -= delta
                
                elif self.text_pooler == 'logits':
                    text_features = outputs.logits
                    

        if projection:
            text_features = self.text_projection_head(text_features)

        return text_features
    
    def forward(self, images, texts, text_only=False):
        """
        images: torch.tensor (batchs_size, preprocessed image)
        texts:  torch.tensor (batchs_size, token_indexs)
        """
        text_features = self.encode_text(texts, projection=True)

        if text_only: # skip image forward for efficient teacher caching 
            image_features = text_features
        else:
            image_features = self.encode_image(images, projection=True)
        
        text_embed = self.proj_text(self.text_hidden)
        image_embed = self.proj_image(self.image_backbone(images, return_hidden=True))
        # print(a.shape)
        # print([text_embed.shape, image_embed.shape])
        # image_o_features = F.normalize(torch.mean(image_embed, dim=1).to(self.device), dim=-1)
        # text_o_features = F.normalize(torch.mean(text_embed, dim=1).to(self.device), dim=-1)

        it_seq = torch.cat([image_embed, text_embed], dim = 1).to(self.device)
        i_len = image_embed.size()[1]
        # text_masks = (texts != 0).long().to(self.device) 
        text_masks = torch.ones(text_embed.size()[:-1],dtype=torch.long).to(self.device)
        image_masks = torch.ones(image_embed.size()[:-1],dtype=torch.long).to(self.device)
        # it_masks = torch.ones(it_seq.size()[:-1],dtype=torch.long).to(self.device)

        att_embed = self.it_encoder(it_seq)

        image_res_embed = self.fc_image(image_embed - att_embed[:,:i_len,:])
        text_res_embed = self.fc_text(text_embed - att_embed[:,i_len:,:])

        image_output = self.fusion_model(image_res_embed, image_masks, text_embed, text_masks)
        text_output = self.fusion_model(text_res_embed, text_masks, image_embed, image_masks)

        image_rec_features = F.normalize(torch.mean(image_output, dim=1), dim=-1)
        text_rec_features = F.normalize(torch.mean(text_output, dim=1), dim=-1)

        # negative_image_features = self.clip_model.encode_image(negative_images) if negative_images is not None else None
        # negative_text_features = self.clip_model.encode_text(negative_texts) if negative_texts is not None else None

        # 计算损失
        # info_nce_loss = self.info_nce_loss(image_features, text_features)
        # negative_image_features = F.normalize(negative_image_features, dim=-1)
        # negative_image_features = F.normalize(negative_image_features, dim=-1)
        # triplet_loss = self.triplet_loss(image_features, text_features, negative_image_features, negative_text_features)

        # rec_loss = self.rec_loss(image_output, image_embed) + self.rec_loss(text_output, text_embed)
        rec_loss = self.rec_loss(image_rec_features, F.normalize(image_features, dim=-1)) +  self.rec_loss(text_rec_features, F.normalize(text_features, dim=-1))
        rec_loss2 = self.info_nce_loss(image_rec_features, text_rec_features)
        # print(image_features.size(), image_o_features.size(), text_features.size(), text_o_features.size())
        
        # total_loss = 0.1 * info_nce_loss + triplet_loss + rec_los
        self.total_loss = rec_loss # + rec_loss2 # + triplet_loss + rec_loss 
        self.total_loss = self.total_loss.to(self.device)

        return image_features, text_features, self.logit_scale.exp(), self.total_loss
        
    def get_loss(self):
        return self.total_loss
        
def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class FusionModel(nn.Module):
    def __init__(self, pretrained_path=None):
        super(FusionModel, self).__init__()
        self.fusion_config = BertConfig(
            hidden_size=512, 
            num_hidden_layers=8,
            num_attention_heads=8,
            is_decoder=True,
            add_cross_attention=True
        )
        # self.fusion_config.hidden_dropout_prob = 0.4  # 设置隐藏层的dropout率
        # self.fusion_config.attention_probs_dropout_prob = 0.4  # 设置注意力层的dropout率
        self.fusion_model = BertModel(config=self.fusion_config)
        if pretrained_path:
            pretrained_dict = torch.load(pretrained_path, map_location='cuda')
            self.fusion_model.load_state_dict(pretrained_dict)

    def forward(self, text_embed, text_masks, image_embed, image_masks):
        fusion_output = self.fusion_model(inputs_embeds=text_embed,
                                          attention_mask=text_masks,
                                          encoder_hidden_states=image_embed,
                                          encoder_attention_mask = image_masks,
                                          return_dict=True)
        fusion_output = fusion_output.last_hidden_state
        return fusion_output

class BiDirectionalInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(BiDirectionalInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        logits_per_image = image_features @ text_features.t() / self.temperature
        logits_per_text = text_features @ image_features.t() / self.temperature
        
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size).to(image_features.device)
        
        loss_i2t = self.cross_entropy_loss(logits_per_image, labels)
        loss_t2i = self.cross_entropy_loss(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2

class BiDirectionalTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(BiDirectionalTripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, image_features, text_features, negative_image_features, negative_text_features):
        loss_i2t = self.triplet_loss(image_features, text_features, negative_text_features)
        loss_t2i = self.triplet_loss(text_features, image_features, negative_image_features)
        return (loss_i2t + loss_t2i) / 2

class GroupedBiDirectionalInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5, group_size=5):
        super(GroupedBiDirectionalInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.group_size = group_size

    def forward(self, image_features, text_features):

        # print(image_features[1]-image_features[0])
        batch_size = image_features.size(0)
        assert batch_size % self.group_size == 0, "Batch size 必须是 group_size 的整数倍"

        logits_per_image = image_features @ text_features.t() / self.temperature
        logits_per_text = text_features @ image_features.t() / self.temperature

        labels = torch.arange(batch_size).to(image_features.device) // self.group_size  # (batch_size,)

        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # (batch_size, batch_size)

        # 计算 softmax，屏蔽掉组内负样本
        exp_logits_per_image = torch.exp(logits_per_image) * positive_mask.float()
        exp_logits_per_text = torch.exp(logits_per_text) * positive_mask.float()

        # 归一化 loss 计算
        loss_i2t = -torch.sum(logits_per_image * positive_mask, dim=1) + torch.log(torch.sum(exp_logits_per_image, dim=1))
        loss_t2i = -torch.sum(logits_per_text * positive_mask, dim=1) + torch.log(torch.sum(exp_logits_per_text, dim=1))

        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2
        return loss

