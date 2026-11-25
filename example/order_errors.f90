program order_errors
 use sklearn,only:dp,median_absolute_error,median_absolute_error_raw,max_error
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2),w(3)
 y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp];w=[1._dp,4._dp,1._dp]
 a(:,1)=y;a(:,2)=y;b(:,1)=p;b(:,2)=y
 print *, 'Median:',median_absolute_error(y,p)
 print *, 'Weighted median:',median_absolute_error(y,p,sample_weight=w)
 print *, 'Per-output weighted median:',median_absolute_error_raw(a,b,sample_weight=w)
 print *, 'Custom-output median:',median_absolute_error(a,b,sample_weight=w,multioutput=[1._dp,3._dp])
 print *, 'Maximum:',max_error(reshape(y,[3,1]),p)
end program
