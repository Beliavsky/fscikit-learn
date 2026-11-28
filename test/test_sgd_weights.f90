program test_sgd_weights
  use sklearn_common,only:dp,require
  use sklearn_sgd,only:sgd_regressor,sgd_classifier
  use sklearn_perceptron,only:perceptron
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  type(sgd_regressor)::r
  type(sgd_classifier)::c
  type(perceptron)::pc
  type(csr_matrix)::s
  real(dp)::one(1,1),two(2,1),x(40,2),y(40),weights(40)
  real(dp),allocatable::data(:),target(:),w(:),scores(:,:),prediction(:)
  integer,allocatable::ptr(:),indices(:),labels(:)
  integer::i,j
  character(40)::mode
  one=1;two(:,1)=[1._dp,-1._dp]
  call get_command_argument(1,mode)
  r%penalty='none';r%learning_rate='constant';r%eta0=.1_dp;r%fit_intercept=.false.;r%max_iter=3;r%tol=-1
  c%penalty='none';c%learning_rate='constant';c%eta0=.25_dp;c%fit_intercept=.false.;c%max_iter=3;c%tol=-1
  pc%eta0=.25_dp;pc%fit_intercept=.false.;pc%max_iter=3;pc%tol=-1;pc%shuffle=.false.
  if(index(mode,'memory_')==1)then
    allocate(ptr(257),indices(512),data(512),target(256),w(256),labels(256))
    do i=1,256
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,500000-i];data(2*i-1:2*i)=[1._dp/(i+1),.5_dp]
      target(i)=sin(real(i,dp));labels(i)=mod(i,3);w(i)=merge(-.25_dp,1._dp,mod(i,4)==0)
    end do
    ptr(257)=513;s=csr_from_arrays(256,500000,ptr,indices,data)
    select case(trim(mode))
    case('memory_reg')
      r%average=1;r%early_stopping=.true.;r%validation_fraction=.25_dp
      call r%fit(s,target,w);r%early_stopping=.false.;call r%partial_fit(s,target,w);prediction=r%predict(s)
      call require(r%t_==1025.and.all(ieee_is_finite(prediction)),'wide signed regression')
    case('memory_clf')
      c%average=1;c%early_stopping=.true.;c%validation_fraction=.25_dp
      call c%fit(s,labels,w,class_weight=[0._dp,-.5_dp,1._dp]);c%early_stopping=.false.
      call c%partial_fit(s,labels,sample_weight=w,class_weight=[0._dp,-.5_dp,1._dp]);scores=c%decision_function(s)
      call require(c%t_==1025.and.all(ieee_is_finite(scores)),'wide signed classifier')
    case('memory_pc')
      pc%early_stopping=.true.;pc%validation_fraction=.25_dp
      call pc%fit(s,labels,w,class_weight=[0._dp,-.5_dp,1._dp]);pc%early_stopping=.false.
      call pc%partial_fit(s,labels,sample_weight=w,class_weight=[0._dp,-.5_dp,1._dp]);scores=pc%decision_function(s)
      call require(pc%t_==1025.and.all(ieee_is_finite(scores)),'wide signed Perceptron')
    case default
      call require(.false.,'unknown signed-weight memory case')
    end select
    print *,'Large signed-weight CSR memory case: PASS';stop
  end if
  do i=1,40
    x(i,:)=[real(i,dp)/40,sin(real(i,dp))];y(i)=x(i,1)-x(i,2)
  end do
  allocate(labels(40));labels=mod([(i,i=1,40)],2);weights=1
  r%early_stopping=.true.;c%early_stopping=.true.;pc%early_stopping=.true.
  r%validation_fraction=.3_dp;c%validation_fraction=.3_dp;pc%validation_fraction=.3_dp
  select case(trim(mode))
  case('reg_nonpositive_validation','clf_nonpositive_validation','pc_nonpositive_validation')
    weights=-1
  case('reg_zero_validation','clf_zero_validation','pc_zero_validation')
    if(index(mode,'reg_')==1)then
      call r%fit(x,y);j=0
      do i=1,40
        if(.not.r%validation_mask_(i))cycle
        j=j+1;weights(i)=merge(1._dp,-1._dp,mod(j,2)==1)
      end do
    else
      call c%fit(x,labels);j=0
      do i=1,40
        if(.not.c%validation_mask_(i))cycle
        j=j+1;weights(i)=merge(1._dp,-1._dp,mod(j,2)==1)
      end do
    end if
  case('class_weight_nan');call c%fit(x,labels,class_weight=[ieee_value(0._dp,ieee_quiet_nan),1._dp])
  end select
  if(index(mode,'reg_')==1)call r%fit(x,y,weights)
  if(index(mode,'clf_')==1)call c%fit(x,labels,weights)
  if(index(mode,'pc_')==1)call pc%fit(x,labels,weights)
  call require(len_trim(mode)==0,'signed-weight guard unexpectedly accepted')
  r%early_stopping=.false.;c%early_stopping=.false.;pc%early_stopping=.false.
  call r%partial_fit(one,[1._dp],[-1._dp])
  call require(abs(r%coef_(1)+.1_dp)<1.e-14_dp,'negative regression sample weight reverses update')
  call r%partial_fit(one,[1._dp],[-1._dp])
  call require(abs(r%coef_(1)+.21_dp)<1.e-14_dp.and.r%t_==3,'signed regression stream clock')
  s=csr_from_dense(two)
  call c%partial_fit(two,[1,-1],classes=[-1,1],sample_weight=[-2._dp,1._dp],class_weight=[-1._dp,0._dp])
  call pc%partial_fit(s,[1,-1],classes=[-1,1],sample_weight=[-2._dp,1._dp],class_weight=[-1._dp,0._dp])
  call require(abs(c%coef_(1,1)+.25_dp)<1.e-14_dp.and.all(c%coef_==pc%coef_),'signed sample/class products')
  print *,'SGD/Perceptron signed sample/class updates, zeros and streams: PASS'
end program
