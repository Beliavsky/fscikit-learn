program reference_pa_seeded_driver
  use sklearn_common,only:dp
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  implicit none
  type(passive_aggressive_regressor)::r
  type(passive_aggressive_classifier)::c
  real(dp),allocatable::x(:,:),q(:,:),y(:),weights(:),value(:,:),prediction(:)
  integer,allocatable::labels(:)
  integer::cases,case_id,n,p,m,family,mode,shuffle,bias,average,seed,weight_mode,limit,patience
  real(dp)::cost,epsilon,tol
  read(*,*)cases
  do case_id=1,cases
    read(*,*)n,p,m,family,mode,shuffle,bias,average,seed,weight_mode,cost,epsilon,tol,limit,patience
    allocate(x(n,p),q(m,p),y(n),labels(n))
    read(*,*)x;read(*,*)q;read(*,*)y
    r=passive_aggressive_regressor();c=passive_aggressive_classifier()
    r%c=cost;r%epsilon=epsilon;r%tol=tol;r%max_iter=limit;r%n_iter_no_change=patience
    r%shuffle=shuffle/=0;r%fit_intercept=bias/=0;r%average=average;r%random_state=seed
    c%c=cost;c%tol=tol;c%max_iter=limit;c%n_iter_no_change=patience
    c%shuffle=shuffle/=0;c%fit_intercept=bias/=0;c%average=average;c%random_state=seed
    print *,'BEGIN',case_id
    if(family==0)then
      if(mode/=0)r%loss='squared_epsilon_insensitive'
      call r%fit(x,y)
      call emit('coef',r%coef_);call emit('intercept',[r%intercept_])
      call emit('train',r%predict(x));call emit('query',r%predict(q))
      call emit('iterations',[real(r%n_iter_,dp)]);call emit('t',[r%t_])
      call emit('features',[real(r%n_features_in_,dp)]);call emit('score',[r%score(x,y)])
    else
      labels=nint(y)
      if(mode/=0)c%loss='squared_hinge'
      c%class_weight_balanced=weight_mode==1
      if(weight_mode>=2)then
        allocate(weights(family));read(*,*)weights
        call c%fit(x,labels,weights);deallocate(weights)
      else
        call c%fit(x,labels)
      end if
      call emit('coef',reshape(c%coef_,[size(c%coef_)]));call emit('intercept',c%intercept_)
      value=c%decision_function(x);call emit('train',reshape(value,[size(value)]))
      value=c%decision_function(q);call emit('query',reshape(value,[size(value)]))
      call emit('prediction',real(c%predict(q),dp));call emit('classes',real(c%classes_,dp))
      call emit('iterations',[real(c%n_iter_,dp)]);call emit('t',[c%t_])
      call emit('features',[real(c%n_features_in_,dp)]);call emit('score',[c%score(x,labels)])
    end if
    print *,'END'
    deallocate(x,q,y,labels)
  end do
contains
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    print '( *(es25.16e3,1x))',values
  end subroutine
end program
