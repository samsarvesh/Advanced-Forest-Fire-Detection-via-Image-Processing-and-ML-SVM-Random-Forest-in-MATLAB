function forest_fire_gui_ml_full_v3()
    % --- Smart Forest Fire Detection with Extended ML Evaluation ---
    % Includes: SVM, Random Forest, Accuracy, Precision, Recall, F1, ROC & AUC

    % ================= FIGURE SETUP =================
    hFig = figure('Name','🔥 Forest Fire Detection — SmartVision AI (ML+ROC)','NumberTitle','off','MenuBar','none','Toolbar','none',...
        'Color',[0.1 0.1 0.1],'Position',[200 100 1200 700]);
    movegui(hFig,'center');

    % ================= AXES =================
    axOrig = axes('Parent',hFig,'Units','pixels','Position',[40 200 520 440]);
    axis(axOrig,'off'); title(axOrig,'Original Image','Color','w','FontSize',11,'FontWeight','bold');
    axRes = axes('Parent',hFig,'Units','pixels','Position',[640 200 520 440]);
    axis(axRes,'off'); title(axRes,'Detection Result','Color','w','FontSize',11,'FontWeight','bold');

    % ================= UI ELEMENTS =================
    uicontrol('Style','text','String','FOREST FIRE DETECTION SYSTEM (ML + ROC + AUC)',...
        'Position',[0 650 1200 40],'FontSize',14,'FontWeight','bold','BackgroundColor',[0.2 0.2 0.2],'ForegroundColor','y');

    uicontrol('Style','text','Position',[25 20 1150 160],'BackgroundColor',[0.15 0.15 0.15]);

    uicontrol('Style','pushbutton','String','📂 Load Image','Position',[50 100 150 40],...
        'FontSize',10,'BackgroundColor',[0.3 0.6 1],'ForegroundColor','w','FontWeight','bold','Callback',@onLoad);

    uicontrol('Style','pushbutton','String','✨ Enhance','Position',[220 100 150 40],...
        'FontSize',10,'BackgroundColor',[0.2 0.8 0.4],'ForegroundColor','w','FontWeight','bold','Callback',@onEnhance);

    uicontrol('Style','pushbutton','String','🔥 Detect Fire','Position',[390 100 150 40],...
        'FontSize',10,'BackgroundColor',[1 0.4 0.2],'ForegroundColor','w','FontWeight','bold','Callback',@onDetect);

    uicontrol('Style','pushbutton','String','🤖 Train + Evaluate ML','Position',[560 100 180 40],...
        'FontSize',10,'BackgroundColor',[0.6 0.4 1],'ForegroundColor','w','FontWeight','bold','Callback',@onTrain);

    uicontrol('Style','pushbutton','String','💾 Save Result','Position',[760 100 150 40],...
        'FontSize',10,'BackgroundColor',[0.4 0.7 1],'ForegroundColor','w','FontWeight','bold','Callback',@onSave);

    uicontrol('Style','pushbutton','String','🔄 Reset','Position',[930 100 150 40],...
        'FontSize',10,'BackgroundColor',[0.8 0.3 0.3],'ForegroundColor','w','FontWeight','bold','Callback',@onReset);

    uicontrol('Style','text','Position',[50 60 140 20],'String','Sensitivity','ForegroundColor','w','BackgroundColor',[0.15 0.15 0.15]);
    sldSens = uicontrol('Style','slider','Min',0.1,'Max',1,'Value',0.5,'Position',[200 60 200 20],'Callback',@livePreview);

    uicontrol('Style','text','Position',[430 60 140 20],'String','Min Fire Area','ForegroundColor','w','BackgroundColor',[0.15 0.15 0.15]);
    edtMinArea = uicontrol('Style','edit','String','300','Position',[560 60 80 25]);

    hStatus = uicontrol('Style','text','Position',[50 20 1100 30],'String','Status: Ready — Load an image.',...
        'ForegroundColor','w','BackgroundColor',[0.1 0.1 0.1],'HorizontalAlignment','left');

    % ================= DATA STORAGE =================
    data = struct('orig',[],'result',[],'mask',[],'svmModel',[],'rfModel',[],...
        'hStatus',hStatus,'axOrig',axOrig,'axRes',axRes,'sldSens',sldSens,'edtMinArea',edtMinArea);
    guidata(hFig,data);

    % ================= CALLBACKS =================
    function onLoad(~,~)
        [fname,fpath]=uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff','Image Files'});
        if isequal(fname,0), return; end
        img = imread(fullfile(fpath,fname));
        axes(axOrig); imshow(img); title(axOrig,'Original Image','Color','w');
        data = guidata(hFig); data.orig = img; guidata(hFig,data);
        info = imfinfo(fullfile(fpath,fname));
        set(hStatus,'String',sprintf('✅ Loaded %s | %dx%d | %.2f KB',fname,info.Width,info.Height,info.FileSize/1024));
    end

    function onEnhance(~,~)
        data = guidata(hFig);
        if isempty(data.orig), set(hStatus,'String','⚠️ Load image first.'); return; end
        img = im2uint8(mat2gray(data.orig));
        for i=1:size(img,3), img(:,:,i)=adapthisteq(img(:,:,i)); end
        axes(axOrig); imshow(img); title(axOrig,'Enhanced Image','Color','w');
        data.orig = img; guidata(hFig,data);
        set(hStatus,'String','✨ Enhanced with adaptive histogram equalization.');
    end

    function onDetect(~,~)
        data = guidata(hFig);
        if isempty(data.orig), set(hStatus,'String','⚠️ Load image first.'); return; end
        [blended,firePercent,nRegions] = detectFire(data.orig);
        axes(axRes); imshow(blended);
        title(axRes,sprintf('Detected Fire: %.2f%%',firePercent),'Color','w');
        data.result = blended; guidata(hFig,data);
        set(hStatus,'String',sprintf('🔥 Detection done. %d regions | %.2f%% fire area.',nRegions,firePercent));
    end

    function onTrain(~,~)
        set(hStatus,'String','🧠 Training ML models... please wait.');
        pause(0.2);
        [X,Y] = buildPatchDataset();
        if isempty(X)
            set(hStatus,'String','⚠️ Dataset missing: plantvillage/fire and /no fire');
            return;
        end

        % --- Safe dataset split ---
        n = numel(Y);
        if n < 5
            warning('⚠️ Too few samples, using all for training & testing.');
            Xtrain=X;Ytrain=Y;Xtest=X;Ytest=Y;
        else
            cv=cvpartition(Y,'HoldOut',0.2);
            Xtrain=X(training(cv),:); Ytrain=Y(training(cv),:);
            Xtest=X(test(cv),:);     Ytest=Y(test(cv),:);
        end

        % --- Train models ---
        svmModel=fitcsvm(Xtrain,Ytrain,'KernelFunction','rbf','Standardize',true,'Probability',true);
        rfModel=TreeBagger(60,Xtrain,Ytrain,'Method','classification');

        % --- Predict + Probability for ROC ---
        [YpredSVM,svmScore]=predict(svmModel,Xtest);
        [YpredRF,rfScore]=predict(rfModel,Xtest);
        YpredRF=str2double(YpredRF);
        rfScore = str2double(rfScore(:,2));

        % --- Metrics ---
        [precSVM,recSVM,f1SVM]=calcMetrics(Ytest,YpredSVM);
        [precRF,recRF,f1RF]=calcMetrics(Ytest,YpredRF);
        accSVM=mean(YpredSVM==Ytest)*100; accRF=mean(YpredRF==Ytest)*100;

        % --- ROC & AUC ---
        [xSVM,ySVM,~,aucSVM]=perfcurve(Ytest,svmScore(:,2),1);
        [xRF,yRF,~,aucRF]=perfcurve(Ytest,rfScore,1);

        % --- Plot ROC curves ---
        figure('Name','ROC Curves — SVM vs RF');
        plot(xSVM,ySVM,'LineWidth',2); hold on;
        plot(xRF,yRF,'LineWidth',2);
        plot([0 1],[0 1],'k--');
        legend({'SVM',['RF (AUC=' num2str(aucRF,'%.2f') ')']},'Location','SouthEast');
        title('ROC Curves for SVM & Random Forest'); xlabel('False Positive Rate'); ylabel('True Positive Rate'); grid on;

        % --- Confusion matrices ---
        figure('Name','Confusion Matrix - SVM');
        confusionchart(Ytest,YpredSVM,'Title','SVM Confusion Matrix');
        figure('Name','Confusion Matrix - Random Forest');
        confusionchart(Ytest,YpredRF,'Title','Random Forest Confusion Matrix');

        % --- Display in console & GUI ---
        msg=sprintf(['✅ ML Evaluation Complete\n' ...
            'SVM → Acc: %.2f%% | Prec: %.2f | Rec: %.2f | F1: %.2f | AUC: %.2f\n' ...
            'RF  → Acc: %.2f%% | Prec: %.2f | Rec: %.2f | F1: %.2f | AUC: %.2f'], ...
            accSVM,precSVM,recSVM,f1SVM,aucSVM,accRF,precRF,recRF,f1RF,aucRF);

        disp(msg);
        set(hStatus,'String',msg);

        % Save models in memory
        data=guidata(hFig);
        data.svmModel=svmModel; data.rfModel=rfModel;
        guidata(hFig,data);
    end

    function [p,r,f1]=calcMetrics(yTrue,yPred)
        tp=sum((yTrue==1)&(yPred==1));
        fp=sum((yTrue==0)&(yPred==1));
        fn=sum((yTrue==1)&(yPred==0));
        p=tp/(tp+fp+eps);
        r=tp/(tp+fn+eps);
        f1=2*p*r/(p+r+eps);
    end

    function onSave(~,~)
        data=guidata(hFig);
        if isempty(data.result), set(hStatus,'String','⚠️ No result to save!'); return; end
        [fname,fpath]=uiputfile('*.png','Save Result');
        if isequal(fname,0),return;end
        imwrite(data.result,fullfile(fpath,fname));
        set(hStatus,'String',sprintf('💾 Saved as %s',fname));
    end

    function onReset(~,~)
        cla(axOrig); cla(axRes);
        title(axOrig,'Original Image','Color','w');
        title(axRes,'Detection Result','Color','w');
        data.orig=[]; data.result=[]; guidata(hFig,data);
        set(hStatus,'String','Status: Ready — Load an image.');
    end

    function livePreview(~,~)
        data=guidata(hFig);
        if isempty(data.orig),return;end
        [blended,~,~]=detectFire(data.orig);
        axes(axRes);imshow(blended);title(axRes,'Live Sensitivity Preview','Color','w');
    end

    % ================= DETECTION CORE =================
    function [blended,firePercent,nRegions]=detectFire(img)
        data=guidata(hFig);
        sens=get(data.sldSens,'Value');
        minArea=str2double(get(data.edtMinArea,'String'));
        if isnan(minArea),minArea=300;end

        hsv=rgb2hsv(img);H=hsv(:,:,1);S=hsv(:,:,2);V=hsv(:,:,3);
        mask=(H>=0&H<=0.15)&(S>0.4*sens)&(V>0.5*sens);
        R=double(img(:,:,1));G=double(img(:,:,2));B=double(img(:,:,3));
        redDom=(R>1.1*G)&(R>1.1*B);
        mask=mask|redDom;mask=imfill(mask,'holes');mask=bwareaopen(mask,round(minArea));
        cc=bwconncomp(mask);stats=regionprops(cc,'Area','BoundingBox');
        totalFirePixels=sum([stats.Area]);totalPixels=numel(mask);
        firePercent=totalFirePixels/totalPixels*100;nRegions=numel(stats);
        overlay=img;overlay(:,:,2)=overlay(:,:,2)*0.3;overlay(:,:,3)=overlay(:,:,3)*0.3;
        overlay(:,:,1)=uint8(double(overlay(:,:,1))+120*double(mask));
        alpha=0.6;blended=uint8(alpha*double(overlay)+(1-alpha)*double(img));
        for k=1:numel(stats)
            blended=insertShape(blended,'Rectangle',stats(k).BoundingBox,'Color','red','LineWidth',3);
        end
    end

    % ================= DATASET BUILDER =================
    function [X,Y]=buildPatchDataset()
        root=fullfile(pwd,'plantvillage');
        fireFolder=fullfile(root,'fire');
        nofireFolder=fullfile(root,'no fire');
        if ~isfolder(fireFolder)||~isfolder(nofireFolder)
            X=[];Y=[];return;
        end
        fireImgs=listImagesInFolder(fireFolder);
        nofireImgs=listImagesInFolder(nofireFolder);
        X=[];Y=[];
        for i=1:numel(fireImgs)
            I=imread(fullfile(fireFolder,fireImgs{i}));
            feat=extractSimpleFeatures(I);
            X=[X;feat];Y=[Y;1];
        end
        for i=1:numel(nofireImgs)
            I=imread(fullfile(nofireFolder,nofireImgs{i}));
            feat=extractSimpleFeatures(I);
            X=[X;feat];Y=[Y;0];
        end
        disp(['✅ Loaded ',num2str(numel(Y)),' samples']);
    end

    function f=extractSimpleFeatures(I)
        I=imresize(I,[128 128]);
        hsv=rgb2hsv(I);
        meanH=mean(hsv(:,:,1),'all');meanS=mean(hsv(:,:,2),'all');
        meanV=mean(hsv(:,:,3),'all');stdV=std2(hsv(:,:,3));
        R=I(:,:,1);G=I(:,:,2);B=I(:,:,3);
        rgRatio=mean(R(:)./(G(:)+1));rbRatio=mean(R(:)./(B(:)+1));
        f=[meanH meanS meanV stdV rgRatio rbRatio];
    end

    function files=listImagesInFolder(folder)
        exts={'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};
        files={};
        for k=1:numel(exts)
            d=dir(fullfile(folder,exts{k}));
            for i=1:numel(d), files{end+1}=d(i).name; end %#ok<AGROW>
        end
    end
end
