program reference_perceptron_driver
  use sklearn_common,only:dp
  use sklearn_perceptron,only:perceptron
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(perceptron)::c
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
    c=perceptron()
    c%penalty=penalty;c%alpha=alpha;c%eta0=eta;c%tol=tol
    c%validation_fraction=fraction;c%max_iter=limit;c%n_iter_no_change=patience
    c%early_stopping=early/=0;c%random_state=seed;c%shuffle=shuffle/=0;c%fit_intercept=bias/=0
    c%warm_start=average==1;c%class_weight_balanced=weight_mode==1
    class_weights=[.5_dp,2._dp,1.3_dp]
    if(weight_mode==2)then
      class_weights=class_weights(:family)
    else
      if(allocated(class_weights))deallocate(class_weights)
    end if
    print *,'BEGIN',case_id
    do step=1,repeats
        call train()
        if(sparse==0)then
          prediction=real(c%predict(x),dp)
        else
          prediction=real(c%predict(sx),dp)
        end if
        call emit('coef',reshape(c%coef_,[size(c%coef_)]));call emit('intercept',c%intercept_)
        call emit('iterations',[real(c%n_iter_,dp)]);call emit('t',[c%t_])
        call emit('mask',real(merge(1,0,c%validation_mask_),dp))
      call emit('classes',real(c%classes_,dp))
      call emit('features',[real(c%n_features_in_,dp)])
      call emit('scores',reshape(c%decision_function(x),[n*size(c%coef_,1)]))
      call emit('score',[c%score(sx,labels,sample_weight=w)])
      call emit('prediction',prediction)
      print *,'STEP'
    end do
    print *,'END'
    deallocate(x,y,w,labels)
  end do
contains
  subroutine train()
    real(dp)::initial(merge(1,family,family==2),p)
    integer::j,k
    do j=1,size(initial,1)
      do k=1,p
        initial(j,k)=.03_dp*(j+k)
      end do
    end do
    if(sparse==0)then
      if(average==2)then
        call c%partial_fit(x,labels,classes=pack([-5,7,21],[(j<=family,j=1,3)]),sample_weight=w,class_weight=class_weights)
      else if(average==3)then
        call c%fit(x,labels,w,class_weights,coef_init=initial,intercept_init=spread(.1_dp,1,size(initial,1)))
      else if(average==4)then
        call c%fit(x,labels,w,class_weights,coef_init=initial(1,:),intercept_init=.1_dp)
      else
        call c%fit(x,labels,w,class_weights)
      end if
    else
      if(average==2)then
        call c%partial_fit(sx,labels,classes=pack([-5,7,21],[(j<=family,j=1,3)]),sample_weight=w,class_weight=class_weights)
      else if(average==3)then
        call c%fit(sx,labels,w,class_weights,coef_init=initial,intercept_init=spread(.1_dp,1,size(initial,1)))
      else if(average==4)then
        call c%fit(sx,labels,w,class_weights,coef_init=initial(1,:),intercept_init=.1_dp)
      else
        call c%fit(sx,labels,w,class_weights)
      end if
    end if
  end subroutine
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    print '( *(es25.16e3,1x))',values
  end subroutine
end program
