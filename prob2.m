t = 0:1/63:1;

y1 = sawtooth(2*pi*(0.5*t+1/4),0.5);    
y2 = 0.5*(sin(2*pi*2*t)+1);
input = [y1;y2];
%% training
learningRate1 = 0.2;
learningRate2 = 2;

nbrOfEpochs_max = 500*64;
nbrOfLayers = 3;
nbrOfNodesPerLayer = [2,16,2];
weights = cell(1, nbrOfLayers);
%hidden_w = cell(1, nbrOfLayers);
Delta_Weights = cell(1, nbrOfLayers);
%%
Samples = input;
num = length(Samples);
%%
for i = 1:length(weights)-1
    weights{i} = 0.1*rand(nbrOfNodesPerLayer(i),nbrOfNodesPerLayer(i+1)) ;
    weights{i}(:,1) = 0; 
    Delta_Weights{i} = zeros(nbrOfNodesPerLayer(i), nbrOfNodesPerLayer(i+1));
end
weights2 = weights;
weights{2}(1,:) = zeros(1,nbrOfNodesPerLayer(i+1));
Delta_hidden_w = zeros(nbrOfNodesPerLayer(2), nbrOfNodesPerLayer(2));
hidden_w = 0.1*rand(nbrOfNodesPerLayer(2),nbrOfNodesPerLayer(2)) ;
hidden_w(:,1) = 0;
%memory length
mem_l = 2;
NodesActivations = cell(1, nbrOfLayers); 
time_lag = 5;
HiddenNodesActivations = cell(1, time_lag);
for i = 1:length(NodesActivations)
    NodesActivations{i} = zeros(1, nbrOfNodesPerLayer(i));
end

HiddenNodesActivations{1} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{2} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{3} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{4} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{5} = zeros(1, nbrOfNodesPerLayer(2));

NodesBackPropagatedErrors = NodesActivations; 
HiddenBackPropagatedErrors = NodesActivations{2}; 
HiddenBackPropagatedErrors2 = NodesActivations{2}; 
HiddenBackPropagatedErrors3 = NodesActivations{2}; 
mse = zeros(1,nbrOfEpochs_max);
training_mse = zeros(1,nbrOfEpochs_max);
train_idx = 1;
testmse = zeros(1,nbrOfEpochs_max);
w_x1 = zeros(nbrOfNodesPerLayer(2),(length(Samples)-1)*nbrOfEpochs_max);
w_didden = zeros(nbrOfNodesPerLayer(2),(length(Samples)-1)*nbrOfEpochs_max);

predicted = zeros(nbrOfEpochs_max*(length(Samples)-1),2);
for ep = 1:nbrOfEpochs_max
    if rem(ep,num-1) == 0
        spl = num-1;
    else
        spl = rem(ep,num-1);
    end
    
    if(ep >100*64)
        xx = NodesActivations{3};
        learningRate1 = 1;
        learningRate2 = 0.01;
    else
        xx = Samples(:,spl)';
    end
    
    NodesActivations{1} = xx;
    NodesActivations{2} = NodesActivations{1}*weights{1} + HiddenNodesActivations{3}*hidden_w ;
    NodesActivations{2} = Activation_func(NodesActivations{2});
    
    HiddenNodesActivations{1} = HiddenNodesActivations{2};
    HiddenNodesActivations{2} = HiddenNodesActivations{3};
    HiddenNodesActivations{3} = HiddenNodesActivations{4};
    HiddenNodesActivations{4} = HiddenNodesActivations{5};
    HiddenNodesActivations{5} = NodesActivations{2};
    
    NodesActivations{3} = NodesActivations{2}*weights{2} ;
    NodesActivations{3} = Activation_func(NodesActivations{3});
    
    gradient = Activation_func_drev(NodesActivations{nbrOfLayers});
    training_mse(train_idx) = sum((Samples(:,spl+1)' - NodesActivations{nbrOfLayers}).^2)/2;
    
    predicted(train_idx,:)= NodesActivations{3};
    
    NodesBackPropagatedErrors{nbrOfLayers} = (Samples(:,spl+1)' - NodesActivations{3}).*gradient;
    train_idx = train_idx+1;
    
    gradient = Activation_func_drev(NodesActivations{2});
    NodesBackPropagatedErrors{2}(1) =  NodesBackPropagatedErrors{3}(1);
    for node=1:length(NodesBackPropagatedErrors{2}) % For all the Nodes in current Layer
        NodesBackPropagatedErrors{2}(node) =  weights{2}(node,:)*(NodesBackPropagatedErrors{3}.*gradient(node))';
    end
    
    gradient_hidden3 = Activation_func_drev(HiddenNodesActivations{2});
    gradient_hidden4 = Activation_func_drev(HiddenNodesActivations{3});
    gradient_hidden5 = Activation_func_drev(HiddenNodesActivations{4});
    
    hidden_w(1,1) = 0;
    %gradient_hidden3(1,:)*
    HiddenBackPropagatedErrors(1) = (hidden_w(node,:)*NodesBackPropagatedErrors{2}')*gradient_hidden5(1) ;
    for node=2:length(HiddenBackPropagatedErrors)
        hidden_w(node,node) = 0;
        HiddenBackPropagatedErrors(node) = hidden_w(node,:)*(NodesBackPropagatedErrors{2}*gradient_hidden5(node))';
    end
    
    HiddenBackPropagatedErrors2(1) = (hidden_w(node,:)*HiddenBackPropagatedErrors')*gradient_hidden4(1) ;
    %HiddenBackPropagatedErrors(1)*gradient_hidden4(1) ;
    for node=2:length(HiddenBackPropagatedErrors2)
        hidden_w(node,node) = 0;
        HiddenBackPropagatedErrors2(node) = hidden_w(node,:)*(HiddenBackPropagatedErrors2*gradient_hidden4(node))';
    end
    
    HiddenBackPropagatedErrors3(1) = (hidden_w(node,:)*HiddenBackPropagatedErrors2') ;
    %HiddenBackPropagatedErrors2(1)*gradient_hidden3(1) ;
    for node=1:length(HiddenBackPropagatedErrors3)
        hidden_w(node,node) = 0;
        HiddenBackPropagatedErrors3(node) = hidden_w(node,:)*(HiddenBackPropagatedErrors3.*gradient_hidden3(node))';
    end
    
    Delta_Weights{2} = learningRate1*NodesActivations{2}'*NodesBackPropagatedErrors{3};
    Delta_Weights{2}(1,:) = (learningRate1*NodesBackPropagatedErrors{3});
    
    Delta_Weights{1} = learningRate2*NodesActivations{1}'*NodesBackPropagatedErrors{2};
    
    Delta_hidden_w = learningRate2*(HiddenBackPropagatedErrors'*HiddenNodesActivations{3}+ HiddenBackPropagatedErrors2'*HiddenNodesActivations{2}+HiddenBackPropagatedErrors3'*HiddenNodesActivations{1});
    Delta_hidden_w(1,:) = learningRate2*(HiddenBackPropagatedErrors+HiddenBackPropagatedErrors2+HiddenBackPropagatedErrors3);
    for node=1:length(HiddenBackPropagatedErrors)
        Delta_hidden_w(node,node) = 0;
    end
    for Layer = 1:nbrOfLayers-1
        weights{Layer} = weights{Layer} + Delta_Weights{Layer};
    end
    %weights{1}(:,1)
    %Delta_Weights{1}(:,1)
    
    hidden_w = hidden_w + Delta_hidden_w;
    w_x1(:,train_idx) = weights{1}(2,:)';
    w_didden(:,train_idx) = hidden_w(:,3);
end

%%
figure(2)
plot(training_mse);
figure(3)
plot(predicted(:,1),predicted(:,2));
%%
% figure(5)
% legendInfo = cell(1);
% wc = w_didden;
% [nr,nc] = size(wc);
% for i=1:nr
% plot(1:nc,wc(i,:),'color',rand(1,3));
% legendInfo{i} = ['weights ' num2str(i)]; 
% hold on
% end
% legend(legendInfo);
% hold off
% xlabel('iterations');
% ylabel('hidden weights');
%%
NodesActivations = cell(1, nbrOfLayers); 
HiddenNodesActivations = cell(1, time_lag);
for i = 1:length(NodesActivations)
    NodesActivations{i} = zeros(1, nbrOfNodesPerLayer(i));
end

HiddenNodesActivations{1} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{2} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{3} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{4} = zeros(1, nbrOfNodesPerLayer(2));
HiddenNodesActivations{5} = zeros(1, nbrOfNodesPerLayer(2));

[output_pred] = test(NodesActivations, HiddenNodesActivations, weights, hidden_w );


figure(4)
plot(output_pred(2:end,1),output_pred(2:end,2));