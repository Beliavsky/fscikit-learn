program test_gpr_multioutput
  use sklearn_gaussian_process, only: gaussian_process_regressor
  use sklearn_gp_kernels
  use sklearn_common, only: dp,require
  implicit none
  type(gaussian_process_regressor) :: model,single
  real(dp) :: x(20,1),y(20,2),weights(20),value,up,down
  real(dp), allocatable :: predicted(:,:),std(:,:),cov(:,:,:),gradient(:),theta(:),point(:)
  integer :: i,j
  do i=1,20
    x(i,1)=-2+4*real(i-1,dp)/19
  end do
  y(:,1)=sin(x(:,1));y(:,2)=3;weights=1;weights(1)=0
  model%kernel=1.2_dp*rbf_covariance(.8_dp)+white_covariance(.02_dp)
  model%alpha_values=.01_dp+.02_dp*cos(x(:,1))**2;model%normalize_y=.true.
  call model%fit(x,y);predicted=model%predict_multi(x);std=model%predict_std_multi(x);cov=model%predict_cov_multi(x)
  call require(all(shape(predicted)==[20,2]),'GPR matrix-target prediction shape')
  call require(model%n_targets_==2.and.size(model%alpha_multi_,2)==2,'GPR target metadata')
  call require(.not.allocated(model%alpha_),'GPR matrix targets do not expose misleading vector coefficients')
  call require(maxval(abs(predicted(:,2)-3))==0,'GPR constant target normalization')
  call require(model%score(x,y,weights)>.99_dp,'GPR weighted multioutput score')
  do j=1,2
    do i=1,20
      call require(abs(std(i,j)**2-cov(i,i,j))<1.e-12_dp,'GPR target-wise covariance scaling')
    end do
  end do
  theta=model%kernel_%theta();value=model%log_marginal_likelihood(theta,gradient)
  call require(abs(value-model%log_marginal_likelihood_)<1.e-12_dp,'GPR cached likelihood')
  do j=1,size(theta)
    point=theta;point(j)=point(j)+1.e-5_dp;up=model%log_marginal_likelihood(point)
    point=theta;point(j)=point(j)-1.e-5_dp;down=model%log_marginal_likelihood(point)
    call require(abs((up-down)/2.e-5_dp-gradient(j))<1.e-6_dp,'GPR joint-target likelihood gradient')
  end do
  call require(all(model%kernel_%theta()==theta),'GPR default likelihood evaluation clones its kernel')
  value=model%log_marginal_likelihood(theta+.03_dp,clone_kernel=.false.)
  call require(maxval(abs(model%kernel_%theta()-theta-.03_dp))<1.e-14_dp,'GPR mutable likelihood evaluation')
  single%kernel=rbf_covariance();single%alpha_values=[.04_dp];call single%fit(x,y(:,:1))
  call require(allocated(single%alpha_),'GPR one-column targets retain vector prediction compatibility')
  predicted=single%predict_multi(x)
  call require(maxval(abs(single%predict(x)-predicted(:,1)))==0,'GPR one-column vector API')
  single%alpha_=0
  call require(all(single%predict(x)==0),'GPR single-target vector coefficients remain authoritative')
  single%normalize_y=.true.;call single%fit(x(:11,:),[(3.2_dp,i=1,11)])
  call require(single%y_train_std_==1.and.all(single%y_train_==0),'GPR NumPy scalar reduction for constant targets')
  print *, 'GPR multiple targets and likelihood gradients: PASS'
end program
