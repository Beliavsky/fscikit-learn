program reference_sgd_validation_driver
  use sklearn_common,only:dp
  use sklearn_sgd,only:sgd_regressor,sgd_classifier
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(sgd_regressor)::r
  type(sgd_classifier)::c
  type(csr_matrix)::sx
  real(dp),allocatable::x(:,:),y(:),w(:),prediction(:),class_weights(:)
  integer,allocatable::labels(:)
  integer::cases,case_id,n,p,family,sparse,seed,average,shuffle,bias,repeats,step,early,weight_mode
  character(32)::loss,penalty,rate
  real(dp)::alpha,eta,tol,fraction
  integer::limit,patience
  read(*,*)cases
  do case_id=1,cases
    read(*,*)n,p,family,sparse,seed,average,shuffle,bias,repeats,early,weight_mode,loss,penalty,rate,alpha,eta,tol,fraction,limit,patience
    allocate(x(n,p),y(n),w(n),labels(n));read(*,*)x;read(*,*)y;read(*,*)w
    labels=0
    if(family/=0)labels=nint(y)
    sx=csr_from_dense(x)
    r=sgd_regressor();c=sgd_classifier()
    r%loss=loss;r%penalty=penalty;r%learning_rate=rate;r%alpha=alpha;r%eta0=eta;r%tol=tol
    r%validation_fraction=fraction;r%max_iter=limit;r%n_iter_no_change=patience
    r%early_stopping=early/=0;r%random_state=seed;r%average=average;r%shuffle=shuffle/=0;r%fit_intercept=bias/=0
    r%warm_start=repeats>1
    c%loss=loss;c%penalty=penalty;c%learning_rate=rate;c%alpha=alpha;c%eta0=eta;c%tol=tol
    c%validation_fraction=fraction;c%max_iter=limit;c%n_iter_no_change=patience
    c%early_stopping=early/=0;c%random_state=seed;c%average=average;c%shuffle=shuffle/=0;c%fit_intercept=bias/=0
    c%warm_start=repeats>1;c%class_weight_balanced=weight_mode==1
    class_weights=[.5_dp,2._dp,1.3_dp]
    if(weight_mode==2)then
      class_weights=class_weights(:family)
    else
      if(allocated(class_weights))deallocate(class_weights)
    end if
    print *,'BEGIN',case_id
    do step=1,repeats
      if(family==0)then
        if(sparse==0)then
          call r%fit(x,y,w);prediction=r%predict(x)
        else
          call r%fit(sx,y,w);prediction=r%predict(sx)
        end if
        call emit('coef',r%coef_);call emit('intercept',[r%intercept_])
        call emit('iterations',[real(r%n_iter_,dp)]);call emit('t',[r%t_])
        call emit('mask',real(merge(1,0,r%validation_mask_),dp))
      else
        if(sparse==0)then
          call c%fit(x,labels,w,class_weight=class_weights);prediction=real(c%predict(x),dp)
        else
          call c%fit(sx,labels,w,class_weight=class_weights);prediction=real(c%predict(sx),dp)
        end if
        call emit('coef',reshape(c%coef_,[size(c%coef_)]));call emit('intercept',c%intercept_)
        call emit('iterations',[real(c%n_iter_,dp)]);call emit('t',[c%t_])
        call emit('mask',real(merge(1,0,c%validation_mask_),dp))
      end if
      call emit('prediction',prediction)
      print *,'STEP'
    end do
    print *,'END'
    deallocate(x,y,w,labels)
  end do
contains
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    print '( *(es25.16e3,1x))',values
  end subroutine
end program
