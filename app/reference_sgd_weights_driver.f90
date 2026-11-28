program reference_sgd_weights_driver
  use sklearn_common,only:dp
  use sklearn_sgd,only:sgd_regressor,sgd_classifier
  use sklearn_perceptron,only:perceptron
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(sgd_regressor)::r
  type(sgd_classifier)::c
  type(perceptron)::pc
  type(csr_matrix)::sx
  real(dp),allocatable::x(:,:),y(:),w(:),prediction(:),class_weights(:),cw_input(:)
  integer,allocatable::labels(:)
  integer::cases,case_id,n,p,family,sparse,seed,average,shuffle,bias,repeats,step,early,weight_mode
  character(32)::loss,penalty,rate
  real(dp)::alpha,eta,tol,fraction
  integer::limit,patience,workflow
  read(*,*)cases
  do case_id=1,cases
    read(*,*)n,p,family,sparse,seed,average,shuffle,bias,repeats,early,weight_mode,loss,penalty,rate,alpha,eta,tol,fraction,limit,patience,workflow
    allocate(x(n,p),y(n),w(n),labels(n));read(*,*)x;read(*,*)y;read(*,*)w
    if(weight_mode==2)then
      allocate(cw_input(family));read(*,*)cw_input
    end if
    labels=0
    if(family/=0)labels=nint(y)
    sx=csr_from_dense(x)
    r=sgd_regressor();c=sgd_classifier();pc=perceptron()
    r%loss=loss;r%penalty=penalty;r%learning_rate=rate;r%alpha=alpha;r%eta0=eta;r%tol=tol
    r%validation_fraction=fraction;r%max_iter=limit;r%n_iter_no_change=patience
    r%early_stopping=early/=0;r%random_state=seed;r%average=average;r%shuffle=shuffle/=0;r%fit_intercept=bias/=0
    r%warm_start=workflow==1
    c%loss=loss;c%penalty=penalty;c%learning_rate=rate;c%alpha=alpha;c%eta0=eta;c%tol=tol
    c%validation_fraction=fraction;c%max_iter=limit;c%n_iter_no_change=patience
    c%early_stopping=early/=0;c%random_state=seed;c%average=average;c%shuffle=shuffle/=0;c%fit_intercept=bias/=0
    c%warm_start=workflow==1;c%class_weight_balanced=weight_mode==1
    pc%penalty=penalty;pc%alpha=alpha;pc%eta0=eta;pc%tol=tol
    pc%validation_fraction=fraction;pc%max_iter=limit;pc%n_iter_no_change=patience
    pc%early_stopping=early/=0;pc%random_state=seed;pc%shuffle=shuffle/=0;pc%fit_intercept=bias/=0
    pc%warm_start=workflow==1;pc%class_weight_balanced=weight_mode==1
    class_weights=[.5_dp,2._dp,1.3_dp]
    if(weight_mode==2)then
      class_weights=cw_input;deallocate(cw_input)
    else
      if(allocated(class_weights))deallocate(class_weights)
    end if
    print *,'BEGIN',case_id
    do step=1,repeats
      if(family==0)then
        if(sparse==0)then
          if(workflow==2)then
            call r%partial_fit(x,y,w)
          else
            call r%fit(x,y,w)
          end if
          prediction=r%predict(x)
        else
          if(workflow==2)then
            call r%partial_fit(sx,y,w)
          else
            call r%fit(sx,y,w)
          end if
          prediction=r%predict(sx)
        end if
        call emit('coef',r%coef_);call emit('intercept',[r%intercept_])
        call emit('iterations',[real(r%n_iter_,dp)]);call emit('t',[r%t_])
        call emit('mask',real(merge(1,0,r%validation_mask_),dp))
      else if(rate=='fixed')then
        if(sparse==0)then
          if(workflow==2)then
            call pc%partial_fit(x,labels,classes=unique_classes(),sample_weight=w,class_weight=class_weights)
          else
            call pc%fit(x,labels,w,class_weight=class_weights)
          end if
          prediction=real(pc%predict(x),dp)
        else
          if(workflow==2)then
            call pc%partial_fit(sx,labels,classes=unique_classes(),sample_weight=w,class_weight=class_weights)
          else
            call pc%fit(sx,labels,w,class_weight=class_weights)
          end if
          prediction=real(pc%predict(sx),dp)
        end if
        call emit('coef',reshape(pc%coef_,[size(pc%coef_)]));call emit('intercept',pc%intercept_)
        call emit('iterations',[real(pc%n_iter_,dp)]);call emit('t',[pc%t_])
        call emit('mask',real(merge(1,0,pc%validation_mask_),dp))
      else
        if(sparse==0)then
          if(workflow==2)then
            call c%partial_fit(x,labels,classes=unique_classes(),sample_weight=w,class_weight=class_weights)
          else
            call c%fit(x,labels,w,class_weight=class_weights)
          end if
          prediction=real(c%predict(x),dp)
        else
          if(workflow==2)then
            call c%partial_fit(sx,labels,classes=unique_classes(),sample_weight=w,class_weight=class_weights)
          else
            call c%fit(sx,labels,w,class_weight=class_weights)
          end if
          prediction=real(c%predict(sx),dp)
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
  function unique_classes()result(values)
    integer::values(family)
    integer::all_labels(3)=[-5,7,21]
    values=all_labels(:family)
  end function
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    print '( *(es25.16e3,1x))',values
  end subroutine
end program
