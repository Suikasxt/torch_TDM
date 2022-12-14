import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import pickle
import faiss
import time
import numpy as np
import os
import sys
sys.path.append('./VQ_DR/deezer')
from utils import get_model
import metrics
from VQ_DR.deezer.utils import generate_data_by_user, print_result


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='YoutubeDNN', type=str, help='GRU4Rec | YoutubeDNN |')
parser.add_argument('--model_type', default='two_tower', type=str, help='VQ or two_tower')
parser.add_argument('--mode', default='train', type=str, help='train|test')
parser.add_argument('--dataset', default='deezer', type=str)
parser.add_argument('--seq_len', default=20, type=int)
parser.add_argument('--batch_size', default=4096, type=int)
parser.add_argument('--inference_batch_size', default=8192, type=int)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--token_embedding_size', default=128, type=int)
parser.add_argument('--l2', default=1e-6, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--sparse_bias', default=1000, type=float)
parser.add_argument('--score_bias', default=None, type=float, help='the scores bias for controlling sparsity '
                                                                    'of scores.')
parser.add_argument('--epoch', default=4000, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--topk', default=[1, 5, 10, 20, 50], nargs='+')
parser.add_argument('--neg_num', default=10, type=int)
parser.add_argument('--patience', default=5, type=int, help='early stop')
parser.add_argument('--monitor', default='recall', type=str, help='[recall,ndcg,hit_ratio]')
parser.add_argument('--debug', default=0, type=int, help='1 or 0')
parser.add_argument('--K_c', default=256, type=int, help='')
parser.add_argument('--K_u', default=4, type=int, help='')
parser.add_argument('--K_i', default=4, type=int, help='')
parser.add_argument('--save_model_by_epoch', default=0, type=int, help='')
parser.add_argument('--num_reported', default=10, type=int)
parser.add_argument('--num_codebooks', default=2, type=int)
parser.add_argument('--att_layer', default=3, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--beta', default=1., type=float, help='the weight of vq_loss')
parser.add_argument('--token_dist', default=False, type=bool, help='token distribution')
parser.add_argument('--pretrain_epoch', default=0, type=int, help='pretrain models without vq')
parser.add_argument('--best_model_pth', default=None, type=str, help='the path of the best trained model.')
args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() and args.gpu != -1 else torch.device(
    'cpu')
args.topk = [int(x) for x in args.topk]
print(args)

def inference(model, data_loader):
    model = model.eval()
    result = {'hit_ratio': np.zeros(len(args.topk)), 'ndcg': np.zeros(len(args.topk)),
              'recall': np.zeros(len(args.topk)), 'precision': np.zeros(len(args.topk))}
    item_embedding = model.get_item_embedding().detach().cpu().numpy()  # (I,D)
    max_topk = max(args.topk)
    res = faiss.StandardGpuResources()  # ????????????GPU
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = args.gpu  # args.faiss_gpu
    index_flat = faiss.GpuIndexFlatIP(res, args.embedding_size, flat_config)
    index_flat.add(item_embedding)
    count = 0
    time_count = 0
    t1 = time.time()
    for user_features, target_items in tqdm(data_loader, "Inference"):
        time_count -= time.time()
        count += len(user_features)
        user_features = torch.stack(user_features, dim=0)
        user_features = user_features.to(device)
        user_embedding = model.user_encoder(user_features).detach().cpu().numpy()
        
        ## below is faiss
        D, I = index_flat.search(user_embedding, k=max_topk)
        
        ## below is numpy
        #score = np.dot(user_embedding, item_embedding.T)
        #I = np.argsort(-score, axis=1)[:, :max_topk]
        time_count += time.time()
        for i in range(len(user_embedding)):
            r = []
            for idx, t in enumerate(I[i]):
                if t in target_items[i]:
                    r.append(1)
                else:
                    r.append(0)

            for j, K in enumerate(args.topk):
                result['hit_ratio'][j] += metrics.hit_at_k(r, K)
                result['ndcg'][j] += metrics.ndcg_at_k(r, K, target_items[i])
                result['recall'][j] += metrics.recall_at_k(r, K, len(target_items[i]))
                result['precision'][j] += metrics.precision_at_k(r, K)
    t2 = time.time()
    result['hit_ratio'] /= count
    result['ndcg'] /= count
    result['recall'] /= count
    result['precision'] /= count
    result['inference_time'] = t2 - t1
    result['search_time'] = time_count
    print_result(result, args.topk)
    return result

def get_dataset():
    if args.dataset == 'deezer':
        data_path = './VQ_DR/deezer/data/deezer'
        song_embedding_pth = os.path.join(data_path, 'song_embedding.npy')
        train_songs_pth = os.path.join(data_path, 'train_songs.pkl')
        train_features_pth = os.path.join(data_path, 'train_feature.pkl')
        valid_songs_pth = os.path.join(data_path, 'valid_songs.pkl')
        valid_features_pth = os.path.join(data_path, 'valid_feature.pkl')
        test_songs_pth = os.path.join(data_path, 'test_songs.pkl')
        test_features_pth = os.path.join(data_path, 'test_feature.pkl')
        data_dir = song_embedding_pth, train_songs_pth, train_features_pth, valid_songs_pth, valid_features_pth, \
                   test_songs_pth, test_features_pth
        return data_dir
    else:
        print("dataset is not exist!!")
        return None

def train(model, train_loader, valid_loader, test_loader):
    print("The Number of Model Paramters:{}".format(sum(p.numel() for p in model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    print("Begin to Train.")
    early_stop = 0
    best_result = {'hit_ratio': np.zeros(len(args.topk)), 'ndcg': np.zeros(len(args.topk)),
                   'recall': np.zeros(len(args.topk)), 'precision': np.zeros(len(args.topk))}
    best_loss = np.inf
    best_model_path = os.path.join('./checkpoint', '_'.join(
        [args.model, args.dataset, 'bz', str(args.batch_size), 'best', str(np.random.randint(99999))]) + '.pth')
    print("The best model will be saved at {}".format(best_model_path))
    for epoch in range(1, args.epoch + 1):
        model.train()
        total_loss, total_vq_loss = 0.0, 0.0
        count = 0
        for (user_features, target_items, neg_items) in tqdm(train_loader, f"Training at epoch {epoch}"):
            input_dict = {'user_features': user_features.to(device),
                          'pos_items': target_items.reshape(-1).to(device),
                          'neg_items': neg_items.reshape(-1, args.neg_num).to(device)}
            loss = model(input_dict)
            total_loss += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
        if str(args.model_type).upper() == 'VQ' or str(args.model_type).upper() == 'DVQ' or str(
                args.model_type).upper() == 'SDVQ':
            print("Epoch: {} Loss:{:.6f} VQ_Loss:{:.6f}".format(epoch, total_loss / count, total_vq_loss / count))
        else:
            print("Epoch: {} Loss:{:.6f}".format(epoch, total_loss / count))
        if args.save_model_by_epoch != 0 and epoch % args.save_model_by_epoch == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_model_path.replace('.pth', '_epoch_{}.pth'.format(epoch)))
        if epoch % args.num_reported == 0:
            if args.monitor == 'loss':
                '''
                    ??????loss????????????early stop
                '''
                if total_loss <= best_loss:
                    best_loss = total_loss
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, best_model_path)
                    print("The best model is saved at {}, and best loss is {:4f}".format(best_model_path, best_loss))
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= args.patience:
                    break
            else:
                '''
                    ??????????????????????????????early stop
                '''
                result = inference(model, valid_loader)
                print('')
                if result[args.monitor][-1] > best_result[args.monitor][-1]:
                    best_result = result
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, best_model_path)
                    print("The best model is saved at {}".format(best_model_path))
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= args.patience:
                    break
                
    print("=" * 28 + 'Best Result' + "=" * 28)
    print_result(best_result, args.topk)
    print("Loading best model at {}".format(best_model_path))
    print("=" * 32 + 'Test' + "=" * 32)
    checkpoint = torch.load(os.path.abspath(best_model_path))
    model.load_state_dict(checkpoint['net'])
        
    result = inference(model, test_loader)
    
    
def main():
    data_dir = get_dataset()
    train_loader, valid_loader, test_loader, song_embedding, n_user, n_item, user_input_dim = generate_data_by_user(
        dataset=args.dataset, data_dir=data_dir, batch_size=args.batch_size, neg_num=args.neg_num,
        inference_batch_size=args.inference_batch_size)
    model = get_model(n_item, song_embedding, user_input_dim, args)
    #print(song_embedding[0])
    model.device = device
    model.to(device)
    if args.mode == 'train':
        train(model, train_loader, valid_loader, test_loader)
    elif args.mode == 'test':
        if args.best_model_pth:
            best_model_path = args.best_model_pth
            checkpoint = torch.load(os.path.abspath(best_model_path))
            model.load_state_dict(checkpoint['net'])
        inference(model, test_loader)


if __name__ == '__main__':
    main()
