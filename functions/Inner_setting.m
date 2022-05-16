function [inner_func] = inner_setting (param)
    param.inner.eta             =       param.lambda/2;                %The regularization parameter in transformed inner problem
    param.inner.eps             =       param.mu/2;                %The smooth parameter in transformed inner problem

    function  nabla_star = nabla_phi_star(var)
        nabla_star = param.inner.eta*var./sqrt(1+norm(var)^2);
    end

    function nabla = nabla_phi(var)
        nabla   = var./(sqrt(param.inner.eta^2 - norm(var)^2));
    end

    function A_w = opt_A (theta, w)
        order = size(w,1)/param.inner.nFeatures;
        if size(theta,1) == param.inner.nFeatures
            theta = repelem(theta,order,1);   
        end
        A_w  = theta.*w;
    end

    function w_star = opt_A_star(theta,u)
        order = size(u,1)/size(theta,1);       
        theta_hat = repelem (theta,order,1);
        u_star = theta_hat.*u;
        w_star = sum(u_star,2);
    end

    function w = trans_w(x,y,w_star) 
        w   = inv(x'*x/length(y) + param.inner.eps*eye(size(x,2)))*(x'*y/length(y)-w_star);
    end

    function  lip  = Lipschitz(theta)
        Nreal   = 10;
        [P]     = size(theta,1);
        valsup  = -Inf;
        for ii=1:Nreal
            w = randn(P,1);
            w = w/norm(w);
            d =norm(opt_A(theta,w),'fro')^2;
            valsup = max(valsup,norm(opt_A(theta,w),'fro')^2);
        end
        lip =  param.inner.eta/param.inner.eps*valsup;
    end

    inner_func       = struct('nabla_phi_star',@nabla_phi_star,'nabla_phi',@nabla_phi,'opt_A',@opt_A,'opt_A_star',@opt_A_star,'trans_w',@trans_w,'Lipschitz',@Lipschitz);
end