program test_perceptron_controls
  use sklearn_common,only:dp,require
  use sklearn_perceptron,only:perceptron,clone_perceptron_parameters
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(perceptron)::c,copy
  type(csr_matrix)::s
  real(dp)::x(6,2),y(6),weights(6)
  real(dp),allocatable::data(:),scores(:,:)
  integer,allocatable::ptr(:),indices(:),large_labels(:),prediction(:)
  integer::labels(6),i
  character(40)::mode
  call get_command_argument(1,mode)
  x(:,1)=[-2._dp,-1._dp,0._dp,1._dp,2._dp,3._dp];x(:,2)=1
  labels=[-5,-5,-5,7,7,7];weights=1;y=labels;s=csr_from_dense(x)
  c%max_iter=3;c%tol=-1
  select case(trim(mode))
  case('missing_classes');call c%partial_fit(x,labels)
  case('one_class');call c%fit(x,7+0*labels)
  case('changed_classes');call c%partial_fit(x,labels,classes=[-5,7]);call c%partial_fit(x,labels,classes=[-5,9])
  case('unknown_class');call c%partial_fit(x,labels,classes=[-5,9])
  case('early_partial');c%early_stopping=.true.;call c%partial_fit(x,labels,classes=[-5,7])
  case('balanced_partial');c%class_weight_balanced=.true.;call c%partial_fit(x,labels,classes=[-5,7])
  case('bad_penalty');c%penalty='bad';call c%fit(x,labels)
  case('bad_alpha');c%alpha=-1;call c%fit(x,labels)
  case('bad_mix');c%l1_ratio=2;call c%fit(x,labels)
  case('bad_eta');c%eta0=0;call c%fit(x,labels)
  case('bad_tol');c%tol=-.5_dp;call c%fit(x,labels)
  case('bad_seed');c%random_state=-1;call c%fit(x,labels)
  case('bad_fraction');c%validation_fraction=0;call c%fit(x,labels)
  case('bad_weight_size');call c%fit(x,labels,sample_weight=[1._dp])
  case('nan_weight');weights(1)=ieee_value(0._dp,ieee_quiet_nan);call c%fit(x,labels,sample_weight=weights)
  case('bad_init');call c%fit(x,labels,coef_init=[1._dp])
  case('wrong_width');call c%partial_fit(x,labels,classes=[-5,7]);call c%partial_fit(x(:,:1),labels)
  case('negative_weight');weights(1)=-1;call c%fit(x,labels,sample_weight=weights)
    print *,'Accepted signed Perceptron sample weights: PASS';stop
  case('nonpositive_class_weight');call c%fit(x,labels,class_weight=[0._dp,1._dp])
    print *,'Accepted nonpositive Perceptron class weights: PASS';stop
  case('memory')
    allocate(ptr(257),indices(512),data(512),large_labels(256))
    do i=1,256
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,500000-i];data(2*i-1:2*i)=[1._dp/(i+1),.5_dp]
      large_labels(i)=mod(i,3)
    end do
    ptr(257)=513;s=csr_from_arrays(256,500000,ptr,indices,data)
    call c%fit(s,large_labels);prediction=c%predict(s);scores=c%decision_function(s)
    call c%partial_fit(s,large_labels)
    call require(size(prediction)==256.and.all(shape(scores)==[256,3]).and.c%t_==1025,'wide perceptron fit and stream')
    print *,'Large CSR Perceptron memory case: PASS';stop
  end select
  call require(len_trim(mode)==0,'Perceptron guard unexpectedly accepted')
  c%eta0=.5_dp;c%shuffle=.false.
  call c%partial_fit(x(:1,:),labels(:1),classes=[7,-5,7])
  call require(all(c%classes_==[-5,7]).and.c%t_==2,'partial declarations deduplicate and allow absent classes')
  call c%partial_fit(s,labels);call require(c%t_==8.and.c%n_iter_==1,'mixed format incremental clock')
  c%warm_start=.true.;c%early_stopping=.true.;c%validation_fraction=.4_dp
  copy=clone_perceptron_parameters(c)
  call require(.not.allocated(copy%coef_).and.copy%warm_start.and.copy%early_stopping,'parameter-only clone')
  call require(copy%eta0==.5_dp.and.copy%validation_fraction==.4_dp,'clone constructor controls')
  c%early_stopping=.false.;c%warm_start=.false.;call c%fit(x,labels)
  call require(c%score(x,labels)==1.and.c%score(s,labels,sample_weight=weights)==1,'dense/CSR Perceptron queries')
  print *,'Perceptron streaming, queries, controls and parameter clones: PASS'
end program
