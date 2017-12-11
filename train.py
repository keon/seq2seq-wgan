import argparse
import torch
from torch import optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
from models import Encoder, Decoder, Seq2Seq, Discriminator
from utils import load_dataset, to_onehot, enable_gradients, disable_gradients
from logger import VisdomWriter, log_samples


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100000,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lamb', type=float, default=10,
                   help='lambda')
    return p.parse_args()


def grad_penalty(D, real, gen, context, lamb):
    alpha = torch.rand(real.size()).cuda()
    x_hat = alpha * real + ((1 - alpha) * gen).cuda()
    x_hat = Variable(x_hat, requires_grad=True)
    context = Variable(context)
    d_hat = D(x_hat, context)
    ones = torch.ones(d_hat.size()).cuda()
    gradients = grad(outputs=d_hat, inputs=x_hat,
                     grad_outputs=ones, create_graph=True,
                     retain_graph=True, only_inputs=True)[0]
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return penalty


def D_loss(D, G, src, trg, lamb, curriculum):
    src_len = min(curriculum, len(src)-1) + 1
    trg_len = min(curriculum, len(src)-1) + 1
    # with gen
    gen_trg, context = G(src[:src_len], trg[:trg_len])
    d_gen = D(gen_trg, context)
    # with real
    trg = to_onehot(trg, D.vocab_size).type(torch.FloatTensor)[1:trg_len]
    trg = Variable(trg.cuda())
    d_real = D(trg, context)
    # calculate gradient panalty
    penalty = grad_penalty(D, trg.data, gen_trg.data, context.data, lamb)
    loss = d_gen.mean() - d_real.mean() + penalty
    return loss


def G_loss(D, G, src, trg, curriculum):
    src_len = min(curriculum, len(src)-1) + 1
    trg_len = min(curriculum, len(src)-1) + 1
    gen_trg, context = G(src[:src_len], trg[:trg_len])
    loss_g = D(gen_trg, context)
    return -loss_g.mean()


def evaluate(e, model, val_iter, vocab_size, DE, EN, curriculum):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        src_len = min(curriculum, len(src)-1) + 1
        trg_len = min(curriculum, len(src)-1) + 1
        output = model(src[:src_len], trg[:trg_len])[0]
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg[1:trg_len].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.data[0]
        log_samples('./.samples/%d-translation.txt' % e, output, EN)
    return total_loss / len(val_iter)


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    # visdom for plotting
    vis_g = VisdomWriter("Generator Loss",
                         xlabel='Iteration', ylabel='Loss')
    vis_d = VisdomWriter("Negative Discriminator Loss",
                         xlabel='Iteration', ylabel='Loss')

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("de_vocab_size: %d en_vocab_size: %d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    G = Seq2Seq(encoder, decoder).cuda()
    D = Discriminator(en_size, embed_size, hidden_size).cuda()
    optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    # TTUR paper https://arxiv.org/abs/1706.08500

    # pretrained
    # G.load_state_dict(torch.load("./.tmp/21.pt"))

    curriculum = 1
    dis_loss = []
    gen_loss = []
    for e in range(1, args.epochs+1):
        # Training
        for b, batch in enumerate(train_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            src, trg = src.cuda(), trg.cuda()
            # (1) Update D network
            enable_gradients(D)
            disable_gradients(G)
            G.eval()
            D.train()
            # clamp parameters to a cube
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)
            D.zero_grad()
            loss_d = D_loss(D, G, src, trg, args.lamb, curriculum)
            loss_d.backward()
            optimizer_D.step()
            dis_loss.append(loss_d.data[0])
            # (2) Update G network
            if b % 10 == 0:
                enable_gradients(G)
                disable_gradients(D)
                D.eval()
                G.train()
                G.zero_grad()
                loss_g = G_loss(D, G, src, trg, curriculum)
                loss_g.backward()
                optimizer_G.step()
                gen_loss.append(loss_g.data[0])
            # plot losses
            if b % 10 == 0 and b > 1:
                vis_d.update(-loss_d.data[0])
                vis_g.update(loss_g.data[0])
        if e % 10 == 0 and e > 1:
            ce_loss = evaluate(e, G, val_iter, en_size, DE, EN, curriculum)
            print(ce_loss)
        if e % 100 == 0 and e > 1:
            curriculum += 1

        # Validation
        # disable_gradients(G)
        # disable_gradients(D)
        # loss_d, loss_g = 0, 0
        # for b, batch in enumerate(val_iter):
        #     src, len_src = batch.src
        #     trg, len_trg = batch.trg
        #     src, trg = src.cuda(), trg.cuda()
        #     # (1) Validate D
        #     loss_d += D_loss(D, G, src, trg, args.lamb, curriculum)
        #     # (2) Validate G
        #     loss_g += G_loss(D, G, src, trg, curriculum)
        # print("loss_d:", loss_d / len(val_iter),
        #       "loss_g", loss_g / len(val_iter))

        # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     print("[!] saving model...")
        #     if not os.path.isdir(".save"):
        #         os.makedirs(".save")
        #     torch.save(G.state_dict(), './.save/wseq2seq_g_%d.pt' % (i))
        #     torch.save(D.state_dict(), './.save/wseq2seq_d_%d.pt' % (i))
        #     best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
