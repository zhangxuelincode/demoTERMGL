function [] = Structureprocess( Theta )
%(structure probability matrix \theta -> interger matrix only with 0/1)
    [~,ind]=max(Theta,[],2);
    Besttheta = zeros(size(Theta));
    for feature = 1:size(Theta,1)
        Besttheta(feature,ind(feature))=1;
    end
    figure();clf; 
    imagesc(Besttheta); 
    xlabel('Groups','Interpreter','latex','fontsize',10)
    ylabel('Features','Interpreter','latex','fontsize',10)
    title(('Our $\hat{\mathcal{S}}$'),'Interpreter','latex','fontsize',10)
    set(gca,'fontsize',20,'clim',[0 1])
    colormap(flipud(gray))
end



