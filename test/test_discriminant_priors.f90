program test_discriminant_priors
  use sklearn_common, only: dp,require
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis,quadratic_discriminant_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(linear_discriminant_analysis) :: model
  type(quadratic_discriminant_analysis) :: quadratic
  real(dp) :: x(30,3),expected(3,3),delta(3),wide(4,6),projector(6,6),sample(6,6),direction(6)
  integer :: labels(30),i,j,c
  real(dp), allocatable :: prob(:,:)
  do i=1,30
    labels(i)=mod(i,3)
    do j=1,3
      x(i,j)=sin(.37_dp*i*j)+labels(i)*.4_dp*j
    end do
  end do
  model%priors=[0._dp,2._dp,6._dp];model%shrinkage_mode='auto';call model%fit(x,labels)
  call require(model%priors_normalized_.and.all(model%priors_==[0._dp,.25_dp,.75_dp]),'LDA normalizes custom priors')
  call require(all(model%priors==[0._dp,2._dp,6._dp]),'LDA preserves prior input')
  call require(model%reference_class_==2,'automatic LDA chooses first positive prior as score reference')
  prob=model%predict_proba(x)
  call require(all(prob(:,1)==0).and.maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'zero prior excludes a class safely')
  model%solver='lsqr';model%shrinkage_mode='fixed';model%shrinkage=0;call model%fit(x,labels)
  expected=0
  do i=1,30
    c=labels(i)+1;delta=x(i,:)-model%means_(c,:)
    do j=1,3
      expected(:,j)=expected(:,j)+model%priors_(c)*delta*delta(j)/10
    end do
  end do
  call require(maxval(abs(expected-model%covariance_(:,:,1)))<1.e-12_dp,'custom priors weight class covariance')
  model%solver='svd';model%priors=[0._dp,0._dp,1._dp];call model%fit(x,labels)
  prob=model%predict_proba(x)
  call require(all(prob(:,3)==1).and.model%n_components_==0,'one positive prior yields prior-only SVD classifier')
  call require(all(ieee_is_nan(model%explained_variance_ratio_)),'one positive prior SVD energy is undefined')
  quadratic%priors=[1._dp,3._dp,6._dp];quadratic%reg_param=.2_dp;call quadratic%fit(x,labels)
  call require(all(quadratic%priors_==quadratic%priors).and..not.quadratic%priors_normalized_,'QDA retains prior weights')
  deallocate(model%priors);call model%fit(x,labels)
  call require(.not.model%priors_normalized_.and.maxval(abs(model%priors_-1._dp/3))<1.e-15_dp, &
    'deallocating prior input restores empirical fitting')
  direction=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
  wide(1,:)=-direction;wide(2,:)=direction;wide(3,:)=0;wide(4,:)=2*direction
  quadratic%priors=[.25_dp,.75_dp];call quadratic%fit(wide,[1,1,2,2])
  do j=1,6
    sample(:,j)=2*direction*direction(j)
  end do
  projector=(quadratic%covariance_(:,:,1)-.8_dp*sample)/.2_dp
  call require(maxval(abs(matmul(projector,projector)-projector))<1.e-10_dp,'wide QDA keeps an orthogonal thin projector')
  call require(abs(sum([(projector(j,j),j=1,6)])-2)<1.e-10_dp,'wide QDA retains min(samples,features) directions')
  print *, 'Custom LDA and QDA priors: PASS'
end program
