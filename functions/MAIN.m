function [ opt ] = MAIN( y,X,param,thetastar,inner_func)
    opt = struct;
    %% initialization                                     
    [hyperGrad_aux,theta] = Initialization(param);                         %Initialize \vartheta

    [plt]   = Display_init(thetastar,theta,param);                        %Plot the oracle varibale structure

    %% iteration
    for iter=1:param.outer.itermax
          %Pick task randomly
          Batch_I  = randperm(param.inner.nTasks,  param.outer.batchSize);
          %The partial gradient w.r.t. \vartheta 
          [theta_hyper,hyperGrad_aux,step_theta] = HQ_Dual_Hypergradient(X,y,Batch_I,hyperGrad_aux,theta,param,inner_func);
          theta   = Proj_unitSimplex(( theta - step_theta * theta_hyper)')';                %Update \theta 
          Display_refresh( plt,  theta,param);
    end

    %% OUTPUT
    opt.theta = theta;
end
    % Display
function [plt1] = Display_init(thetastar,theta,param)
    if param.outer.displayOnline            
        figure;
        subplot(1,2,1)
        imagesc(thetastar);
        title('Oracle','Interpreter','latex','fontsize',13)
        ylabel('Features','Interpreter','latex','fontsize',2)
        xlabel('Group indices','Interpreter','latex','fontsize',2)
        set(gca,'fontsize',15,'clim',[0 1])
        xticks(1:param.outer.nGroups);
        colormap(flipud(gray))
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        subplot(1,2,2)
        plt1=imagesc(theta);
        title('Ours','Interpreter','latex','fontsize',13)
        ylabel('Features','Interpreter','latex','fontsize',2)
        xlabel('Group indices','Interpreter','latex','fontsize',2)
        set(gca,'fontsize',15,'clim',[0 1])
        xticks(1:param.outer.nGroups);
        colormap(flipud(gray))             
    end
end
function [] = Display_refresh( plt1,theta,param)
    if param.outer.displayOnline
        plt1.CData = theta;
        refreshdata(plt1,'caller');        
        drawnow limitrate
    end
end
    %Initialization
function [hyperGrad_aux, theta] = Initialization (param)
    theta =  Proj_unitSimplex( ((1/param.inner.nGroups)*ones(param.inner.nFeatures,param.inner.nGroups) + .01*randn(param.inner.nFeatures,param.inner.nGroups))')'; 
    hyperGrad_aux.all   = zeros(param.inner.nFeatures,param.inner.nGroups,param.inner.nTasks);
    hyperGrad_aux.mean  = zeros(param.inner.nFeatures,param.inner.nGroups);             
end
