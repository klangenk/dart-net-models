from fastai.vision.all import *
import math
from lib.label import *

class ImageTuple(fastuple):
    @classmethod
    def create(cls, fn):
        image = PILImage.create(fn)
        t = image2tensor(image)
        number_of_imgs = int(t.shape[2] / (t.shape[1] * 1.0))
        s = torch.split(t, t.shape[2] // number_of_imgs, 2)
        images = [PILImage.create(x.permute(1,2,0)) for x in s]
        #categories = TensorMultiCategory(one_hot([vocab.o2i[x] for x in label_field(fn)], 81).float())
        #TensorPoint.create
        return cls(*images)
    
    def show(self, ctx=None, **kwargs): 
        t1,t2 = self
        #title = t3 if isinstance(t3, str) else None
        if not isinstance(t1, Tensor): t1 = image2tensor(t1)
        if not isinstance(t2, Tensor): t2 = image2tensor(t2)
        if t1.shape != t2.shape: return ctx
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        #r = image2tensor(ref.resize((t1.shape[1], t1.shape[1])))
        return show_image(torch.cat([t1,line,t2], dim=2), ctx=ctx, **kwargs)
        #return show_image(torch.cat([r,line, t1,line,t2], dim=2), ctx=ctx, **kwargs)

def ImageTupleBlock(): return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)

def _parent_idxs(items, name):
    print(items)
    def _inner(items, name): return mask2idxs(Path(o).parent.name == name for o in items)
    return [i for n in L(name) for i in _inner(items,n)]

def ParentSplitter(train_name='train', valid_name='valid'):
    "Split `items` from the parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _parent_idxs(o, train_name),_parent_idxs(o, valid_name)
    return _inner


def _idxs(items, name, exclude):
    def _inner(items, name): return mask2idxs(name in Path(o).parts and not any([x in Path(o).parts for x in exclude]) for o in items)
    return [i for n in L(name) for i in _inner(items,n)]

def Splitter(train_name='train', valid_name='valid', exclude = []):
    def _inner(o):
        return _idxs(o, train_name, exclude),_idxs(o, valid_name, exclude)
    return _inner

categories = ['0-1','0-10','0-11','0-12','0-13','0-14','0-15','0-16','0-17','0-18','0-19','0-2','0-20','0-3','0-4','0-5','0-6','0-7','0-8','0-9','0-x1','0-x2','0-x3','0-z25-1','0-z25-2','0-z0','0-zempty','1-1','1-10','1-11','1-12','1-13','1-14','1-15','1-16','1-17','1-18','1-19','1-2','1-20','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-x1','1-x2','1-x3','1-z25-1','1-z25-2','1-z0','1-zempty','2-1','2-10','2-11','2-12','2-13','2-14','2-15','2-16','2-17','2-18','2-19','2-2','2-20','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-x1','2-x2','2-x3','2-z25-1','2-z25-2','2-z0','2-zempty']
vocab = CategoryMap(categories)
c = len(vocab)

def load_data(bs, size, train_name, valid_name, max_rotate=180.0, path = 'data', exclude = []):
    path = Path(path)
    block = DataBlock(
        blocks=(ImageTupleBlock, MultiCategoryBlock(vocab=categories),MultiCategoryBlock(vocab=categories)),
        n_inp=2,
        get_items=get_image_files,
        splitter=Splitter(train_name=train_name, valid_name=valid_name, exclude=exclude),
        item_tfms=Resize(size),
        get_x=[lambda x:x, label_field_random_empty],
        get_y=[label_field],
        batch_tfms=[*aug_transforms(
            do_flip=False,
            max_rotate=max_rotate, 
            max_lighting=0.5,
            pad_mode='zeros'
        ),Normalize.from_stats(*imagenet_stats)]
    )
    return block.dataloaders(path, bs=bs, shuffle=True)

    
