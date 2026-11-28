program reference_pa_validation_driver
  use sklearn_common,only:dp
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(passive_aggressive_regressor)::r
  type(passive_aggressive_classifier)::c
  type(csr_matrix)::sx,sq,original
  real(dp),allocatable::x(:,:),q(:,:),y(:),weights(:),z(:,:)
  integer,allocatable::labels(:),classes(:)
  integer::cases,case_id,family,mode,shuffle,bias,average,seed,weight_mode,steps,p,m,declared,step,fmt,n,action,pass_classes,warm,mutate,limit,patience,early
  real(dp)::cost,epsilon,tolerance,fraction
  read(*,*)cases
  do case_id=1,cases
    read(*,*)family,mode,shuffle,bias,average,seed,weight_mode,cost,epsilon,steps,p,m,declared,limit,patience,tolerance,early,fraction
    allocate(classes(declared),q(m,p));read(*,*)classes;read(*,*)q;call read_csr(sq)
    if(weight_mode>=2)then
      allocate(weights(family));read(*,*)weights
    end if
    r=passive_aggressive_regressor();c=passive_aggressive_classifier()
    r%c=cost;r%epsilon=epsilon;r%random_state=seed;r%shuffle=shuffle/=0;r%fit_intercept=bias/=0
    r%average=average;r%max_iter=limit;r%n_iter_no_change=patience;r%tol=tolerance
    c%c=cost;c%random_state=seed;c%shuffle=shuffle/=0;c%fit_intercept=bias/=0
    c%average=average;c%max_iter=limit;c%n_iter_no_change=patience;c%tol=tolerance;c%class_weight_balanced=weight_mode==1
    if(mode/=0)then
      r%loss='squared_epsilon_insensitive';c%loss='squared_hinge'
    end if
    r%validation_fraction=fraction;c%validation_fraction=fraction
    print *,'BEGIN',case_id
    do step=1,steps
      read(*,*)fmt,n,action,average,pass_classes,warm,mutate
      allocate(y(n),labels(n))
      if(fmt==0)then
        allocate(x(n,p));read(*,*)x
      else
        call read_csr(sx);original=sx
      end if
      read(*,*)y;r%average=average;c%average=average
      r%warm_start=warm/=0;c%warm_start=warm/=0
      r%early_stopping=early/=0.and.action==1;c%early_stopping=early/=0.and.action==1
      if(mutate/=0)then
        if(family==0)then
          r%coef_=2*r%coef_+.1_dp;r%intercept_=.25_dp
        else
          c%coef_=2*c%coef_+.1_dp;c%intercept_=.25_dp
        end if
      end if
      if(family==0)then
        if(fmt==0)then
          if(action==1)then
            call r%fit(x,y)
          else
            call r%partial_fit(x,y)
          end if
          call emit('train',r%predict(x))
        else
          if(action==1)then
            call r%fit(sx,y)
          else
            call r%partial_fit(sx,y)
          end if
          call emit('train',r%predict(sx))
        end if
        call emit('coef',r%coef_);call emit('intercept',[r%intercept_])
        call emit('query_dense',r%predict(q));call emit('query_csr',r%predict(sq))
        call emit('iterations',[real(r%n_iter_,dp)]);call emit('t',[r%t_]);call emit('features',[real(r%n_features_in_,dp)])
        call emit('mask',real(merge(1,0,r%validation_mask_),dp))
      else
        labels=nint(y)
        if(fmt==0)then
          if(action==1)then
            call c%fit(x,labels,class_weight=weights)
          else if(pass_classes/=0)then
            call c%partial_fit(x,labels,classes=classes,class_weight=weights)
          else
            call c%partial_fit(x,labels,class_weight=weights)
          end if
          z=c%decision_function(x)
        else
          if(action==1)then
            call c%fit(sx,labels,class_weight=weights)
          else if(pass_classes/=0)then
            call c%partial_fit(sx,labels,classes=classes,class_weight=weights)
          else
            call c%partial_fit(sx,labels,class_weight=weights)
          end if
          z=c%decision_function(sx)
        end if
        call emit('train',reshape(z,[size(z)]));call emit('coef',reshape(c%coef_,[size(c%coef_)]))
        call emit('intercept',c%intercept_);call emit('classes',real(c%classes_,dp))
        z=c%decision_function(q);call emit('query_dense',reshape(z,[size(z)]))
        z=c%decision_function(sq);call emit('query_csr',reshape(z,[size(z)]))
        call emit('prediction',real(c%predict(sq),dp));call emit('iterations',[real(c%n_iter_,dp)])
        call emit('t',[c%t_]);call emit('features',[real(c%n_features_in_,dp)])
        call emit('mask',real(merge(1,0,c%validation_mask_),dp))
      end if
      if(fmt==1)then
        call emit('preserved',[real(merge(1,0,all(sx%data==original%data).and.all(sx%indices==original%indices).and. &
          all(sx%indptr==original%indptr)),dp)])
      else
        call emit('preserved',[1._dp]);deallocate(x)
      end if
      print *,'STEP'
      deallocate(y,labels)
    end do
    print *,'END'
    deallocate(classes,q)
    if(allocated(weights))deallocate(weights)
  end do
contains
  subroutine read_csr(matrix)
    type(csr_matrix),intent(out)::matrix
    integer::nz
    read(*,*)matrix%n_rows,matrix%n_cols,nz
    allocate(matrix%indptr(matrix%n_rows+1),matrix%indices(nz),matrix%data(nz))
    read(*,*)matrix%indptr
    if(nz>0)then
      read(*,*)matrix%indices;read(*,*)matrix%data
    end if
  end subroutine
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    print '( *(es25.16e3,1x))',values
  end subroutine
end program
