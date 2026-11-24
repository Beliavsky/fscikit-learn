program scale_axes
 use sklearn,only:dp,scale
 implicit none
 real(dp)::x(3,2)
 real(dp),allocatable::columns(:,:),rows(:,:),vector(:)
 logical::centered_again,scaled_again
 x(:,1)=[1._dp,2._dp,3._dp];x(:,2)=[2._dp,4._dp,6._dp]
 columns=scale(x,centering_corrected=centered_again,scaling_corrected=scaled_again)
 rows=scale(x,axis=1)
 vector=scale(x(:,1))
 print *, 'First row after column scaling:',columns(1,:)
 print *, 'First row after row scaling:',rows(1,:)
 print *, 'Vector scaling:',vector
 print *, 'Extra centering/scaling corrections:',centered_again,scaled_again
end program
