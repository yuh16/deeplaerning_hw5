function [predicted] = test(NodesActivations, HiddenNodesActivations, weights, hidden_w )
predicted = zeros(64,2);
output = [0.0, 0.5];
spl = 1;
for i = 1:6400
    NodesActivations{1} = output;
    
    NodesActivations{2} = NodesActivations{1}*weights{1} +HiddenNodesActivations{5}*hidden_w ;
    
    %sum(NodesActivations{1}*weights{1})
    %sum(HiddenNodesActivations{3}*hidden_w)
    %HiddenNodesActivations{4}*hidden_w
    %HiddenNodesActivations{5}*hidden_w
    %disp(2)
    NodesActivations{2} = Activation_func(NodesActivations{2});
    
    HiddenNodesActivations{1} = HiddenNodesActivations{2};
    HiddenNodesActivations{2} = HiddenNodesActivations{3};
    HiddenNodesActivations{3} = HiddenNodesActivations{4};
    HiddenNodesActivations{4} = HiddenNodesActivations{5};
    HiddenNodesActivations{5} = NodesActivations{2};
    
    NodesActivations{3} = NodesActivations{2}*weights{2} ;
    NodesActivations{3} = Activation_func(NodesActivations{3});
    
    output = NodesActivations{3};
    predicted(i,:) = output;
    spl = spl+1; 
end
end
