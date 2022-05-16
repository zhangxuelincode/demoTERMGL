function [err_val,err_tst,width_tst,cover_tst] = Evaluation(x,y,theta,param,inner_func)
 error_val = zeros(1,param.inner.nTasks);
 error_tst = zeros(1,param.inner.nTasks);
 cover = zeros(1,param.inner.nTasks);
 width = zeros(1,param.inner.nTasks);
for xtt=1:param.inner.nTasks
    [output] = HQ_DFBB(x.trn{xtt},y.trn{xtt},theta,inner_func,param); 
    beta = output{end}.w;
    error_val(xtt) =  norm(y.tst{xtt}-x.tst{xtt}*(beta))^2/50;
    error_tst(xtt) =  norm(y.true{xtt}-x.tst{xtt}*(beta))^2/50; 
    [width(xtt),cover(xtt)] =  coverage(y.tst{xtt},x.tst{xtt}*(beta),param.alpha);
end
err_val = sum(error_val)/param.inner.nTasks;
err_tst = sum(error_tst)/param.inner.nTasks;
width_tst = sum(width)/param.inner.nTasks;
cover_tst = sum(cover)/param.inner.nTasks;
end

function [width,cover] = coverage (y,y_pre,alpha)
epsilon = y - y_pre;
N = size(epsilon,1);
epsilon_sort =  sort(epsilon);
n1 = floor(N*alpha/2);
n2 = N - n1;
k1 = n1; 
k2 = n2;
g = ksdensity(epsilon,epsilon_sort);
t=1000;
    while (k1-1)*(k2-N)~=0 
        if g(k1)< g(k2) && g(k1+1)<g(k2+1)
            k1 = k1+1;
            k2 = k2+1;
        end
        if  g(k1)>g(k2) && g(k1-1)>g(k2-1)
 
              k1 = k1-1;
              k2 = k2-1;
        end          
        if t==k1       
            break
        end
        t  = k1;
    end
low =  y_pre + epsilon_sort(k1);
up  =  y_pre + epsilon_sort(k2);

y_low = (y -low)>0;
y_up  = (up-y)>0;
cover = sum(y_low==y_up)/N;
width = abs(epsilon_sort(k2)-epsilon_sort(k1));
end