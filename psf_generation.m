% This programm is designed to simulation coherent and incoherent illumination
% through psf

%% Parameters

lambda = 633e-9; % m
w = 2*pi*3*10 / 6.3;
w1 = 2*pi*3*10/6;
w2 =2*pi*3*10/6.6;
k0 = 2 * pi / lambda;
strength_absorption = 0; % range of random absorption
strength_phase = 2; % range of random phase change


%% Sizes

% object
L_obj = 5e-3; % m
N_obj = 101;
x_obj = linspace(-L_obj / 2, L_obj / 2, N_obj);
dx_obj = x_obj(2) - x_obj(1);
[x_obj, y_obj] = meshgrid(x_obj, x_obj);
% n_pad = (N - N_obj) / 2;

y=0;
data = load('mnist_train.mat');
Image = data.trainImage;
Label = data.trainLabel;
%x=100*rand(1,400)
%y=100*rand(1,400)
%scatter(x,y)
% image
L_img = 5e-3; % m
N_img = 101;
x_img = linspace(-L_img / 2, L_img / 2, N_img);
dx_img = x_img(2) - x_img(1);
[x_img, y_img] = meshgrid(x_img, x_img);
% n_pad = (N - N_img) / 2;
%非偏光stoke矢量的六个分量
    unpolarx = zeros(1, 1);
    unpolary = zeros(1, 1);
    theta=zeros(1, 1);
    unxi=zeros(1, 1);
    unyi=zeros(1, 1);
    untotal=zeros(1, 1);
    un45i=zeros(1, 1);
    un135i=zeros(1, 1);
    unrighti=zeros(1, 1);
    unlefti=zeros(1, 1);
    flagx = 1;
    flagy = 1;
    distance=0;
    t=0;%先设距离，飞行时间等于0，后面再变
    %生成非偏振光
     for n = 1 :10
         %unpolarx(n) = unifrnd(0,0.5)*(exp(-1i * (k0 * distance-unifrnd(w2,w1)*t+unifrnd(0,2*pi))) );
         %unpolary(n) = unifrnd(0,0.5)*(exp(-1i * (k0 * distance-unifrnd(w2,w1)*t+unifrnd(0,2*pi))) );
         unpolarx(n) = 0.5*rand(1,1)*(exp(-1i * (k0 * distance-(w2+(w1-w2)*rand(1,1))*t+2*pi*rand(1,1))) );
         unpolary(n) =  0.5*rand(1,1)*(exp(-1i * (k0 * distance-(w2+(w1-w2)*rand(1,1))*t+2*pi*rand(1,1))) );
         theta(n) = atan(unpolary(n)./unpolarx(n));
         
         unxi(n) = abs(unpolarx(n)).^2;%第n个偏振光x方向光强
         unyi(n) = abs(unpolary(n)).^2;%y方向光强
         un45i(n) =abs(sqrt(2)/2 *unpolarx(n)+sqrt(2)/2 *unpolary(n)).^2;%45方向光强
         un135i(n) =abs(sqrt(2)/2 *unpolarx(n)-sqrt(2)/2 *unpolary(n)).^2;%135方向光强
         unrighti(n) = abs((sqrt(2)/2 *unpolarx(n)*exp(1i*pi/2))+(sqrt(2)/2 *unpolary(n))).^2;%右旋圆
         unlefti(n) = abs((sqrt(2)/2 *unpolarx(n)*exp(-1i*pi/2))+(sqrt(2)/2 *unpolary(n))).^2;
         untotal(n) =unxi(n)+unyi(n);%总光强
         
         
     end
    %max(unxi)
    %compass(unpolarx,unpolary ,'r') 
    %unpolarx(1)
    %unxi(1)
    
    %偏振度计算,应用stoke矢量
    ix=sum(unxi)
    iy=sum(unyi)
    itotal=sum(untotal);
    s0=ix+iy
    %s1=(ix-iy)/s0
    s1=(ix-iy)
    i45=sum(un45i)
    i135=sum(un135i)
    %s2=(i45-i135)/s0
    s2=(i45-i135)
    iright=sum(unrighti)
    ileft=sum(unlefti)
    %s3=(iright-ileft)/s0;
    s3=(iright-ileft)
    DOP=sqrt(s1^2+s2^2+s3^2)/s0
    %把师兄的光加进来，计算偏振度

%% Distances

doi = 10; %m

for depth_position = (1:5)
    d = doi + depth_position;

    % calculate the psf at distance_position
    psf_oix = zeros(N_obj, N_obj, N_img^2);
    psf_oiy = zeros(N_obj, N_obj, N_img^2);


    
    for i = 1 : N_obj
        for j = 1 : N_obj
                xt = x_obj(1, j) - x_img;				
                yt = y_obj(N_obj + 1 - i, 1) - flip(y_img, 1);	
                doi_tmp = sqrt(xt.^2 + yt.^2 + d^2);
                psf_oix(:, :, j+i*N_obj-101) = (exp(-1i * k0 * doi_tmp) ./ (doi_tmp) );
                psf_oiy(:, :, j+i*N_obj-101) = (exp(-1i *k0 * doi_tmp) ./ (doi_tmp) );
                


               
          
                %abs(psf_oix(1, 1, 1));
                %abs(psf_oiy(1, 1, 1));
                %psf_oiy(1, 1, 1);
                %flagx = flagx + 1;
                %flagy = flagy + 1;
        end
    end
    psf_oix = reshape(psf_oix, N_obj^2, N_img^2);
    psf_oiy = reshape(psf_oiy, N_obj^2, N_img^2);
   %调整偏振光振幅，从而达到调节偏振度目的
    psf_oix =1000*psf_oix ;
    psf_oiy = 1000*psf_oiy;
    x=depth_position
    %save(['psf_pix_', num2str(depth_position), '.mat'], 'psf_oix');
    %save(['psf_piy_', num2str(depth_position), '.mat'], 'psf_oiy');
    %原mnist文件，节省硬盘空间放在一起
    coherent_img = zeros(N_img, N_img, 1);
    incoherent_img = zeros(N_img, N_img, 1);
    % p = 1.1 to p=1.9
%    for p = (11:19)
 %       eval(['img_', num2str(p),'= zeros(N_img, N_img, 1)']) ; % img_p = zeros(N_img, N_img, 1)
 %   end

    for pos = (1:500)
        im_org = Image(:,:,pos);
        im_org = double(im_org);
        im_org = imresize(im_org, N_obj / 28, 'nearest');
        im_obj = sqrt(double(im_org));

        
        % p=1 means coherent, p=2 means incoherent, the p value is a real number, change it as you wish.
        % propogation to image (coherent)
        im_imgx = psf_oix.^1 * (reshape(im_obj, N_obj^2, 1).^1);
        im_imgx = reshape(im_imgx, N_img, N_img);
        im_imgy = psf_oiy.^1 * (reshape(im_obj, N_obj^2, 1).^1);
        im_imgy = reshape(im_imgy, N_img, N_img);
        im_img = abs(im_imgx).^2+abs(im_imgy).^2;
        for m =(1:10)
            psf_oixm=unpolarx(m)*ones(N_obj^2);
            im_imgmx=psf_oixm.^1 * (reshape(im_obj, N_obj^2, 1).^1);
            im_imgmx=reshape(im_imgmx, N_img, N_img);
            psf_oiym=unpolary(m)*ones(N_obj^2);
            im_imgmy=psf_oiym.^1 * (reshape(im_obj, N_obj^2, 1).^1);
            im_imgmy=reshape(im_imgmy, N_img, N_img);
            im_img =im_img+abs(im_imgmx).^2+abs(im_imgmy).^2;
        end
            
        y=y+1
        
        
        % incoherent
        %im_img_inc = psf_oi.^2 * (reshape(im_obj, N_obj^2, 1).^2);
        %im_img_inc = reshape(im_img_inc, N_img, N_img);
        %im_img_inc = abs(im_img).^1;

        % Normalization
        min_val = min(min(im_img));
        max_val = max(max(im_img));
        im_img = (im_img - min_val)/(max_val - min_val);
 
        %min_val = min(min(im_img_inc));
        %max_val = max(max(im_img_inc));
        %im_img_inc = (im_img_inc - min_val)/(max_val - min_val);

%         % save
        coherent_img(:,:,pos) = im_img;
        %incoherent_img(:,:,pos) = im_img_inc;

        % The code here is to simulate p=1.1 to p=1.9
        % p = 1.1 to p=1.9
       % for p = (11:19)

            % simulation
        %    temp_img = psf_oi.^(p/10) * (reshape(im_obj, N_obj^2, 1).^(p/10));
        %    temp_img = reshape(temp_img, N_img, N_img);
        %    temp_img = abs(temp_img).^(2/(p/10));
            
            % normalization
         %   min_val = min(min(temp_img));
         %   max_val = max(max(temp_img));
          %  temp_img = (temp_img - min_val)/(max_val - min_val); 
            
            % save
          %  eval(['img_', num2str(p), '(:,:,pos)', '= temp_img']);  % img_p(:,:,pos) = temp_img
        
        index(pos) = Label(pos);

        % print some info every 100 pics.
        if mod(pos, 100) == 0
            pos
        end
    end

    save(['coherent_', num2str(depth_position),'.mat'], 'coherent_img', 'index', '-v7.3')
    %depth_position
end
