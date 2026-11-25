program test_dictionary_functions
  use sklearn_common,only:dp,require
  use sklearn_dictionary,only:dict_learning,dict_learning_online,dictionary_learning_result, &
    online_dictionary_result,dictionary_learning,mini_batch_dictionary_learning
  use,intrinsic::iso_fortran_env,only:int64
  implicit none
  type(dictionary_learning_result)::batch,copy
  type(online_dictionary_result)::online
  type(dictionary_learning)::model
  type(mini_batch_dictionary_learning)::mini
  real(dp)::x(4,2),d(2,2),u(4,2)
  real(dp),allocatable::code(:,:)
  integer::calls,steps
  x(:,1)=[1._dp,2._dp,0._dp,0._dp];x(:,2)=[0._dp,0._dp,1._dp,2._dp]
  d=0;d(1,1)=1;d(2,2)=1;u=0;calls=0;steps=0
  batch=dict_learning(x,2,.1_dp,max_iter=11,tol=0._dp,method='cd',dict_init=d, &
    code_init=u,callback=observe_batch)
  call require(calls==3.and.batch%n_iter==11,'batch callback iterations 0,5,10')
  call require(size(batch%errors)==11,'batch error history retained')
  copy=batch;batch%dictionary=0
  call require(all(copy%dictionary==d),'returned dictionary owns independent storage')
  model%n_components=2;model%alpha=.1_dp;model%max_iter=11;model%tol=0;model%fit_algorithm='cd'
  model%dict_init=d;model%code_init=u;code=model%fit_transform(x)
  call require(maxval(abs(code-copy%code))<1.e-14_dp,'functional batch optimization code matches estimator')
  online=dict_learning_online(x,2,alpha=.1_dp,max_iter=2,dict_init=d,batch_size=2,shuffle=.false., &
    method='cd',tol=0._dp,max_no_improvement=-1,callback=observe_online)
  call require(steps==4.and.online%n_steps==4.and.online%n_iter==2,'online callback and epoch accounting')
  mini%n_components=2;mini%alpha=.1_dp;mini%max_iter=2;mini%dict_init=d;mini%batch_size=2
  mini%shuffle=.false.;mini%fit_algorithm='cd';mini%tol=0;mini%max_no_improvement=-1
  mini%transform_algorithm='lasso_cd';mini%transform_alpha=.1_dp
  call mini%fit(x);code=mini%transform(x)
  call require(maxval(abs(code-online%code))<1.e-14_dp,'online function returns lasso rather than default OMP code')
  online=dict_learning_online(x,2,max_iter=0,dict_init=d,return_code=.false.)
  call require(.not.allocated(online%code).and.all(online%dictionary==d),'dictionary-only zero-epoch call')
  x=0;batch=dict_learning(x,2,.3_dp,max_iter=11,method='cd')
  call require(batch%n_iter==2.and.abs(batch%errors(2)-.6_dp)<1.e-14_dp,'zero-input CD source warm-code behavior')
  call require(batch%code(1,1)==1.and.batch%code(2,2)==1,'zero-input SVD starts with identity code')
  print *,'Dictionary functions, callbacks, estimator consistency and owned results passed'
contains
  subroutine observe_batch(iteration,code,dictionary,errors)
    integer,intent(in)::iteration
    real(dp),intent(inout)::code(:,:),dictionary(:,:),errors(:)
    call require(iteration==5*calls,'callback uses zero-based batch iteration')
    call require(size(errors)==iteration+1,'callback sees current history')
    calls=calls+1
  end subroutine
  subroutine observe_online(step,cost,dictionary,a,b)
    integer(int64),intent(in)::step
    real(dp),intent(in)::cost
    real(dp),intent(inout)::dictionary(:,:),a(:,:),b(:,:)
    call require(step==steps,'callback uses zero-based online step')
    call require(all(shape(a)==[2,2]).and.all(shape(b)==[2,2]),'callback sees online sufficient statistics')
    steps=steps+1
  end subroutine
end program
