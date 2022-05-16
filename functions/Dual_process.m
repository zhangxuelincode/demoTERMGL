function [output] = Dual_process (x_trn,y_trn,theta,inner_func,param)
output = struct;
if ~isfield(param.inner,'stepsize')
    stepsize = .99/inner_func.Lipschitz(theta);
else
    assert(param.inner.stepsize < 1/inner_func.Lipschitz(theta),'Invalid stepsize');
    stepsize = param.inner.stepsize;
end
if ~isfield(param.inner,'initialPoint')
    u = zeros(size(x_trn,2),size(theta,2));
else
    u   = param.initialPoint.u;
end
for iter = 1:param.inner.itermax  
    %%%%%%%%%%%%%%%%% Forward Backward Scheme %%%%%%%%%%%%%%%%%    
    w   = inner_func.trans_w(x_trn,y_trn,inner_func.opt_A_star(theta,u)); %W = (X'X + εI)^-1 * (X'y - Aθ*u)
    Aw  = inner_func.opt_A(theta,w);  % 这是 Aθ 哈达玛积 W
    % Forward step
    v = stepsize*Aw;
    for ll=1:param.inner.nGroups
        v(:,ll) = v(:,ll) + inner_func.nabla_phi(u(:,ll));
    end    
    % Backward step: dual update
    for ll=1:param.inner.nGroups
        u(:,ll) = inner_func.nabla_phi_star(v(:,ll));
    end    
    %%%%%%%%%%%%%%%%%%% Storing and measures %%%%%%%%%%%%%%%%%%
    % Storing: iterates (verify location)
    if param.inner.saveIterates
        output.iterates.w{iter}     = w;
        output.iterates.v{iter}     = v;
        output.iterates.u{iter}     = u;
    end
end
        output.w     = w;
        output.v     = v;
        output.u     = u; 
        output.stepsize = stepsize;
end






