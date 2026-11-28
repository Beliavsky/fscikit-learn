program reference_nusvc_libsvm_driver
  use sklearn_nu_classification,only:nu_svc
  use sklearn_common,only:dp,unique_labels
  implicit none
  type(nu_svc)::model
  real(dp),allocatable::x(:,:),q(:,:),weights(:),cw(:),scores(:,:)
  integer,allocatable::y(:),classes(:)
  integer::cases,c,n,p,m,balanced,shape,ties,gamma_rule,shrinking
  character(32)::mode
  call get_command_argument(1,mode)
  read(*,*)cases
  do c=1,cases
    model=nu_svc();model%shrinking=.true.
    read(*,*)n,p,m,model%kernel,model%nu,model%gamma,model%coef0,model%degree,model%tol, &
      model%max_iter,balanced,shape,ties,gamma_rule,shrinking
    model%shrinking=shrinking/=0
    model%decision_function_shape='ovr';if(shape==0)model%decision_function_shape='ovo'
    model%break_ties=ties/=0;model%gamma_rule='scale';if(gamma_rule==1)model%gamma_rule='auto'
    allocate(x(n,p),q(m,p),weights(n),y(n));read(*,*)x;read(*,*)q;read(*,*)y;read(*,*)weights
    classes=unique_labels(y);allocate(cw(size(classes)));read(*,*)cw
    call model%fit(x,y,weights)
    print *,'BEGIN',c
    call emit('classes',real(model%classes_,dp));call emit('support',real(model%support_-1,dp))
    call emit('n_support',real(model%n_support_,dp));call emit('support_vectors',reshape(model%support_vectors_,[size(model%support_vectors_)]))
    call emit('dual_coef',reshape(model%dual_coef_,[size(model%dual_coef_)]));call emit('intercept',model%intercept_)
    if(allocated(model%coef_))then
      call emit('coef',reshape(model%coef_,[size(model%coef_)]))
    else
      call emit('coef',[real(dp)::])
    end if
    call emit('status',[real(model%fit_status_,dp)])
    call emit('gamma',[model%gamma_])
    scores=model%decision_function(x);call emit('train',reshape(scores,[size(scores)]))
    scores=model%decision_function(q);call emit('query',reshape(scores,[size(scores)]))
    call emit('predict',real(model%predict(q),dp));call emit('score',[model%score(x,y),model%score(x,y,weights)])
    call emit('native_support_original',real(model%support_-1,dp))
    call emit('native_iterations',real(model%n_iter_,dp));call emit('native_kkt',model%kkt_violation_)
    call emit('native_alpha',reshape(model%pair_alpha_,[size(model%pair_alpha_)]))
    call emit('native_scales',model%pair_scales_)
    print *,'END';deallocate(x,q,weights,y,cw)
  end do
contains
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    write(*,'(*(es25.16e3,1x))')values
  end subroutine
end program
