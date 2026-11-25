program kernel_dispatch
 use sklearn,only:dp,pairwise_kernels
 use sklearn_gp_kernels,only:gp_kernel,white_covariance
 implicit none
 real(dp)::x(2,2),y(2,2)
 real(dp),allocatable::k(:,:)
 type(gp_kernel)::white
 x(1,:)=[0._dp,0._dp];x(2,:)=[1._dp,1._dp]
 y(1,:)=[1._dp,0._dp];y(2,:)=[1._dp,1._dp]
 k=pairwise_kernels(x,y)
 print *,'Linear second row: ',k(2,:)
 k=pairwise_kernels(x,y,metric='poly',degree=2)
 print *,'Polynomial second row: ',k(2,:)
 white=white_covariance(.3_dp);k=pairwise_kernels(x,metric=white)
 print *,'White GP dispatch diagonal: ',k(1,1),k(2,2)
end program
