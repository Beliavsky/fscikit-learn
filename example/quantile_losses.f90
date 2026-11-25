program quantile_losses
 use sklearn,only:dp,mean_pinball_loss,d2_pinball_score,d2_absolute_error_score,mean_pinball_loss_raw
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2)
 y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp]
 a(:,1)=y;a(:,2)=y;b(:,1)=p;b(:,2)=y
 print *, 'Pinball:',mean_pinball_loss(y,p)
 print *, 'Pinball D2:',d2_pinball_score(y,p)
 print *, 'Absolute D2:',d2_absolute_error_score(y,p)
 print *, 'Custom pinball:',mean_pinball_loss(a,b,multioutput=[1._dp,3._dp])
 print *, 'Raw pinball:',mean_pinball_loss_raw(a,b)
end program
