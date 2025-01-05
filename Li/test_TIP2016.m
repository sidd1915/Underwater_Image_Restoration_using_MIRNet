% If you use this code, please cite this paper: Chongyi Li, Jichang Guo,Runmin Cong, Yanwei Pang, Bo Wang,
%��Underwater image enhancement by dehazing with minimum information loss and histogram distribution prior��,
%IEEE Transactions on Image Processing, 25(12), pp. 5664-5677 (2016). 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dehazed: underwater image dehazing results
% dehazed_exposure: dehazed results processed by adaptive exposure
% HE_prior: final result by minimum information loss and histrogram distribution prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
% images are returned with absolute path
% add your path
addpath('D:\����\ˮ��ͼ��\����\TIP2016-code�����\Dependencies');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [fn,pn,fi]=uigetfile('*.bmp;*.jpg;*.png','pick up one picture');
% str0='D:\img\OutputImages\';
% save_path=[str0,fn];

imgPath = 'D:\img\InputImages\'; % ͼ���·��
imgDir = dir([imgPath '*.jpg']); % ��������jpg��ʽ�ļ�
tic
for i = 1:length(imgDir) % �����ṹ��Ϳ���һһ����ͼƬ��
    s = strsplit(imgDir(i).name,'.');
    name = [char(s(1)),'_li.jpg'];
    img_input = imread([imgPath imgDir(i).name]); %��ȡÿ��ͼƬ
    fprintf(imgDir(i).name);
    fprintf('\n');
    [dehazed, dehazed_exposure, HE_prior]=master(img_input);
    %figure,imshow([img_input,dehazed,dehazed_exposure,HE_prior]);
    %figure,imshow([img_input,HE_prior]);
    imwrite(HE_prior, fullfile('D:\img\OutputImages\', name));
end
% tic
% img_input=imread([pn fn]);
% [dehazed, dehazed_exposure, HE_prior]=master(img_input);
% %figure,imshow([img_input,dehazed,dehazed_exposure,HE_prior]);
% %figure,imshow([img_input,HE_prior]);
% imwrite(HE_prior,save_path);
mytimer1=toc;
disp(mytimer1)