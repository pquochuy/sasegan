clear all
close all
clc

list_file = dir('../data/clean_testset_wav_16k/*.wav');

% different model checkpoints
cp = {'97000','97100','97200','97300','97400'};
Ncp = numel(cp);

fs_signal = 16000;
% attention layer indes, you may want to change this to the value that you set when training the model
att_layer_index = 2; 

ret = zeros(Ncp, 6);
for c = 1 : Ncp
    ret_c = zeros(numel(list_file),6);
    parfor f = 1 : numel(list_file)
        disp(list_file(f).name);
        clean_wav = ['../data/clean_testset_wav_16k/', list_file(f).name];
        noisy_wav = ['../sasegan/cleaned_testset_wav_16k_att',num2str(att),'_', cp{c}, '/', list_file(f).name];
        spesq = pesq(clean_wav, noisy_wav);
        [~,ssnr] = comp_snr(clean_wav, noisy_wav);
        [Csig,Cbak,Covl] = composite(clean_wav,noisy_wav);

        [x, ~] = audioread(clean_wav);
        [y, ~] = audioread(noisy_wav);
        d_stoi = stoi(x, y, fs_signal);

        ret_c(f,:) = [spesq, Csig, Cbak, Covl, ssnr, d_stoi];
    end
    ret(c, :) = mean(ret_c);
end
disp('Average: PESQ, CSIG, CBAK, COVL, SSNR, STOI')
mean(ret)