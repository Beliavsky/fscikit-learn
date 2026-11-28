program test_pa_partial
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  implicit none
  type(passive_aggressive_regressor)::r
  type(passive_aggressive_classifier)::c,m
  type(csr_matrix)::s,q
  real(dp)::one(1,1),two(1,1),zero(1,1)
  real(dp),allocatable::data(:),y(:),prediction(:),scores(:,:)
  integer,allocatable::ptr(:),indices(:),labels(:)
  integer::i
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(257),indices(512),data(512),y(256),labels(256))
    do i=1,256
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,500000-i]
      data(2*i-1:2*i)=[1._dp/(i+1),.5_dp];y(i)=sin(real(i,dp));labels(i)=mod(i,3)
    end do
    ptr(257)=513;s=csr_from_arrays(256,500000,ptr,indices,data)
    r%average=1;c%average=1
    call r%partial_fit(s,y);call c%partial_fit(s,labels,classes=[0,1,2])
    call r%partial_fit(s,y);call c%partial_fit(s,labels)
    prediction=r%predict(s);scores=c%decision_function(s)
    call require(r%t_==513.and.c%t_==513,'wide partial PA public clocks')
    call require(size(prediction)==256.and.all(shape(scores)==[256,3]),'wide partial PA queries')
    print *,'Large streaming CSR passive-aggressive memory case: PASS';stop
  end if
  one=1;two=2;zero=0;s=csr_from_dense(one);q=csr_from_dense(two)
  select case(trim(mode))
  case('missing_classes');call c%partial_fit(one,[-5])
  case('one_class');call c%partial_fit(one,[-5],classes=[-5])
  case('changed_classes');call c%partial_fit(one,[-5],classes=[-5,7]);call c%partial_fit(one,[7],classes=[-5,9])
  case('unknown_class');call c%partial_fit(one,[9],classes=[-5,7])
  case('reg_width');call r%partial_fit(one,[1._dp]);call r%partial_fit(reshape([1._dp,2._dp],[1,2]),[1._dp])
  case('reg_width_csr');call r%partial_fit(s,[1._dp]);q%n_cols=2;call r%partial_fit(q,[1._dp])
  case('class_width');call c%partial_fit(one,[-5],classes=[-5,7]);call c%partial_fit(reshape([1._dp,2._dp],[1,2]),[7])
  case('class_width_csr');call c%partial_fit(s,[-5],classes=[-5,7]);q%n_cols=2;call c%partial_fit(q,[7])
  case('balanced_first');c%class_weight_balanced=.true.;call c%partial_fit(one,[-5],classes=[-5,7])
  case('balanced_missing');c%class_weight_balanced=.true.;call c%fit(reshape([1._dp,2._dp],[2,1]),[-5,7]);call c%partial_fit(one,[-5])
  case('reg_enable_average');call r%partial_fit(one,[1._dp]);r%average=1;call r%partial_fit(one,[1._dp])
  case('class_enable_average');call c%partial_fit(one,[-5],classes=[-5,7]);c%average=1;call c%partial_fit(one,[7])
  end select
  call require(len_trim(mode)==0,'PA streaming guard unexpectedly accepted')
  r%shuffle=.false.;r%fit_intercept=.false.;r%epsilon=0;r%average=1
  call r%partial_fit(one,[1._dp])
  call require(r%coef_(1)==1.and.r%t_==2.and.r%n_iter_==1,'first one-epoch PA update')
  call r%partial_fit(q,[0._dp])
  call require(abs(r%coef_(1)-.5_dp)<1.e-14_dp.and.r%t_==3,'mixed CSR update retains running average')
  r%c=.2_dp;call r%partial_fit(one,[5._dp])
  call require(abs(r%coef_(1)-.4_dp)<1.e-14_dp.and.r%t_==4,'PA continues standard coefficients behind published average')
  c%c=.25_dp;c%loss='squared_hinge';c%average=1;c%shuffle=.false.
  call c%partial_fit(zero,[-5],classes=[7,-5,7])
  call require(all(c%classes_==[-5,7]).and.abs(c%intercept_(1)+.5_dp)<1.e-14_dp,'class declaration deduplicates and sorts')
  call c%partial_fit(zero,[7]);call require(abs(c%intercept_(1))<1.e-14_dp,'Source binary standard-bias continuation quirk')
  call c%partial_fit(zero,[7]);call require(abs(c%intercept_(1)-1._dp/6)<1.e-14_dp,'Source averaged binary bias over three calls')
  m%c=.25_dp;m%loss='squared_hinge';m%average=1;m%shuffle=.false.
  call m%partial_fit(zero,[-5],classes=[-5,7,21])
  call m%partial_fit(zero,[7])
  call require(maxval(abs(m%intercept_-[-.25_dp,.25_dp,-.75_dp]))<1.e-14_dp, &
    'Source multiclass returned bias overwrites prior averaged publication')
  call m%fit(reshape([1._dp,2._dp],[2,1]),[-4,9]);call m%partial_fit(zero,[-4])
  call require(size(m%classes_)==2.and.all(m%classes_==[-4,9]).and.m%n_iter_==1,'fit resets declared classes and streaming heads')
  r%c=1;r%max_iter=1;r%average=0;r%tol=-1
  call r%fit(one,[1._dp]);call r%partial_fit(q,[0._dp])
  call require(abs(r%coef_(1))<1.e-14_dp.and.r%t_==3,'full fit resets prior streaming averages and clock')
  print *,'PA streaming epochs, standard/average buffers, classes, bias semantics and refits: PASS'
end program
