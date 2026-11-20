! Dense sklearn 1.6.1 decomposition/_nmf.py MiniBatchNMF contracts (BSD-3-Clause).
! Upstream notices: LICENSES/scikit-learn-COPYING, LICENSES/numpy-* and THIRD_PARTY_NOTICES.md.
module sklearn_minibatch_nmf
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_randomized_svd_core, only: randomized_svd
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  private
  public :: mini_batch_nmf
  real(dp), parameter :: eps32=1.1920928955078125e-7_dp
  type :: mini_batch_nmf
    integer :: n_components=0,n_components_=0,batch_size=1024,batch_size_=0,max_iter=200,random_state=0
    integer :: max_no_improvement=10,fresh_restarts_max_iter=30,transform_max_iter=0,transform_max_iter_=0,n_features_in_=0
    integer(int64) :: n_steps_=0,n_iter_=0,no_improvement_=0
    real(dp) :: beta_loss=2,tol=1.e-4_dp,alpha_w=0,alpha_h=-1,l1_ratio=0,forget_factor=.7_dp
    real(dp) :: beta_loss_=2,gamma_=1,rho_=0,reconstruction_err_=0,ewa_cost_=0,ewa_cost_min_=0
    logical :: fresh_restarts=.false.
    character(8) :: init='auto'
    real(dp), allocatable :: components_(:,:),components_numerator_(:,:),components_denominator_(:,:)
  contains
    procedure :: fit
    procedure :: fit_transform
    procedure :: partial_fit
    procedure :: transform
    procedure :: inverse_transform
  end type
contains
  subroutine controls(self)
    class(mini_batch_nmf), intent(in) :: self
    call require(self%n_components>=0.and.self%n_components<=huge(1)-10,'MiniBatchNMF invalid component count')
    call require(self%max_iter>=1.and.self%fresh_restarts_max_iter>=1.and.self%transform_max_iter>=0, &
      'MiniBatchNMF iteration limits must be positive (transform zero means auto)')
    call require(self%batch_size>=1,'MiniBatchNMF batch_size must be positive')
    call require(self%random_state>=0,'MiniBatchNMF seed must be nonnegative')
    call require(self%max_no_improvement==-1.or.self%max_no_improvement>=1,'MiniBatchNMF patience must be positive or -1')
    call require(all(ieee_is_finite([self%beta_loss,self%tol,self%alpha_w,self%alpha_h,self%l1_ratio,self%forget_factor])), &
      'MiniBatchNMF controls must be finite')
    call require(self%tol>=0.and.self%alpha_w>=0.and.(self%alpha_h==-1.or.self%alpha_h>=0), &
      'MiniBatchNMF tolerance/regularization must be nonnegative (alpha_h -1 means same)')
    call require(self%l1_ratio>=0.and.self%l1_ratio<=1,'MiniBatchNMF l1_ratio must be in [0,1]')
    call require(self%forget_factor>=0.and.self%forget_factor<=1,'MiniBatchNMF forget_factor must be in [0,1]')
    call require(self%init=='auto'.or.self%init=='random'.or.self%init=='nndsvd'.or.self%init=='nndsvda'.or. &
      self%init=='nndsvdar'.or.self%init=='custom','unknown MiniBatchNMF initializer')
  end subroutine
  subroutine data_check(x,beta)
    real(dp), intent(in) :: x(:,:),beta
    call check_x(x); call require(all(x>=0),'MiniBatchNMF X must be nonnegative')
    if(beta<=0) call require(all(x>0),'MiniBatchNMF nonpositive beta requires strictly positive X')
  end subroutine
  subroutine query(self,x)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%components_),'MiniBatchNMF is not fitted')
    call data_check(x,self%beta_loss_)
    call require(size(x,2)==self%n_features_in_,'MiniBatchNMF feature count mismatch')
  end subroutine
  subroutine resolve(self,x,h)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: h(:,:)
    self%n_components_=self%n_components
    if(self%n_components==0) then
      self%n_components_=size(x,2)
      if(self%init=='custom'.and.present(h)) self%n_components_=size(h,1)
    end if
    call require(self%n_components_>=1.and.self%n_components_<=huge(1)-10,'MiniBatchNMF invalid component count')
    self%n_features_in_=size(x,2); self%batch_size_=min(self%batch_size,size(x,1)); self%beta_loss_=self%beta_loss
    self%rho_=self%forget_factor**(real(self%batch_size_,dp)/size(x,1))
    self%gamma_=1
    if(self%beta_loss_<1) self%gamma_=1/(2-self%beta_loss_)
    if(self%beta_loss_>2) self%gamma_=1/(self%beta_loss_-1)
    self%transform_max_iter_=self%transform_max_iter
    if(self%transform_max_iter==0) self%transform_max_iter_=self%max_iter
  end subroutine
  subroutine initialize(self,x,w0,h0,w,h)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    real(dp), allocatable, intent(out) :: w(:,:),h(:,:)
    real(dp), allocatable :: u(:,:),s(:),vt(:,:),xp(:),xn(:),yp(:),yn(:)
    real(dp) :: avg,scale,np,nm,sp,sn,mass
    type(legacy_random_state) :: rng
    integer :: n,p,k,i,j
    character(8) :: method
    n=size(x,1); p=size(x,2); k=self%n_components_; method=self%init
    allocate(w(n,k),h(k,p)); w=0; h=0
    if(method=='custom') then
      call require(present(w0).and.present(h0),'MiniBatchNMF custom initialization requires W and H')
      call require(all(shape(w0)==[n,k]).and.all(shape(h0)==[k,p]),'MiniBatchNMF initializer shape mismatch')
      call require(all(ieee_is_finite(w0)).and.all(ieee_is_finite(h0)),'MiniBatchNMF initializer must be finite')
      call require(all(w0>=0).and.all(h0>=0),'MiniBatchNMF initializer must be nonnegative')
      call require(maxval(w0)>0.and.maxval(h0)>0,'MiniBatchNMF initializer must not be all zero')
      w=w0; h=h0; return
    end if
    if(method=='auto') then
      method='nndsvda'
      if(k>min(n,p)) method='random'
    end if
    avg=sum(x)/real(size(x),dp); call rng%seed(self%random_state)
    if(method=='random') then
      scale=sqrt(avg/k)
      do i=1,k
        do j=1,p
          h(i,j)=abs(scale*rng%normal())
        end do
      end do
      do i=1,n
        do j=1,k
          w(i,j)=abs(scale*rng%normal())
        end do
      end do
      return
    end if
    call require(k<=min(n,p),'MiniBatchNMF NNDSVD requires components <= min(samples,features)')
    call randomized_svd(x,k,rng,s,u,vt)
    w(:,1)=sqrt(s(1))*abs(u(:,1)); h(1,:)=sqrt(s(1))*abs(vt(1,:))
    do j=2,k
      xp=max(u(:,j),0._dp); xn=-min(u(:,j),0._dp); yp=max(vt(j,:),0._dp); yn=-min(vt(j,:),0._dp)
      np=sqrt(sum(xp*xp)); nm=sqrt(sum(xn*xn)); sp=sqrt(sum(yp*yp)); sn=sqrt(sum(yn*yn))
      if(np*sp>nm*sn) then
        mass=np*sp
        if(mass>0) then
          scale=sqrt(s(j)*mass); w(:,j)=scale*(xp/np); h(j,:)=scale*(yp/sp)
        end if
      else
        mass=nm*sn
        if(mass>0) then
          scale=sqrt(s(j)*mass); w(:,j)=scale*(xn/nm); h(j,:)=scale*(yn/sn)
        end if
      end if
    end do
    where(w<1.e-6_dp) w=0
    where(h<1.e-6_dp) h=0
    if(method=='nndsvda') then
      where(w==0) w=avg
      where(h==0) h=avg
    else if(method=='nndsvdar') then
      ! Integer seeds are checked twice by source: refill starts from the seed again.
      call rng%seed(self%random_state)
      do i=1,n
        do j=1,k
          if(w(i,j)==0) w(i,j)=abs(avg*rng%normal()/100)
        end do
      end do
      do i=1,k
        do j=1,p
          if(h(i,j)==0) h(i,j)=abs(avg*rng%normal()/100)
        end do
      end do
    end if
  end subroutine
  subroutine regularization(self,n,p,l1w,l2w,l1h,l2h)
    class(mini_batch_nmf), intent(in) :: self
    integer, intent(in) :: n,p
    real(dp), intent(out) :: l1w,l2w,l1h,l2h
    real(dp) :: ah
    ah=self%alpha_h; if(ah==-1) ah=self%alpha_w
    l1w=p*self%alpha_w*self%l1_ratio; l2w=p*self%alpha_w*(1-self%l1_ratio)
    l1h=n*ah*self%l1_ratio; l2h=n*ah*(1-self%l1_ratio)
  end subroutine
  subroutine weights(self,x,w,h,numerator,denominator,hidden)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:),w(:,:),h(:,:)
    real(dp), allocatable, intent(out) :: numerator(:,:),denominator(:,:)
    logical, intent(in) :: hidden
    real(dp), allocatable :: prediction(:,:),safe(:,:),power(:,:),ratio(:,:)
    real(dp) :: beta,l1w,l2w,l1h,l2h
    integer :: n,p,k
    n=size(x,1); p=size(x,2); k=size(h,1); beta=self%beta_loss_
    call regularization(self,n,p,l1w,l2w,l1h,l2h)
    if(beta==2) then
      if(hidden) then
        numerator=matmul(transpose(w),x)
        if(real(k,dp)*k*(real(n,dp)+p)<2._dp*n*k*p) then
          denominator=matmul(matmul(transpose(w),w),h)
        else
          denominator=matmul(transpose(w),matmul(w,h))
        end if
      else
        numerator=matmul(x,transpose(h)); denominator=matmul(w,matmul(h,transpose(h)))
      end if
    else
      prediction=matmul(w,h); safe=prediction; power=prediction
      if(beta<1) power=max(power,eps32)
      if(beta<2) safe=max(safe,eps32)
      if(beta==1) then
        ratio=x/safe
      else if(beta==0) then
        ratio=(1/safe)**2*x
      else
        ratio=safe**(beta-2)*x
      end if
      if(hidden) then
        numerator=matmul(transpose(w),ratio)
        if(beta==1) then
          denominator=spread(sum(w,dim=1),2,p)
          where(denominator==0) denominator=1
        else
          denominator=matmul(transpose(w),power**(beta-1))
        end if
      else
        numerator=matmul(ratio,transpose(h))
        if(beta==1) then
          denominator=spread(sum(h,dim=2),1,n)
        else
          denominator=matmul(power**(beta-1),transpose(h))
        end if
      end if
    end if
    if(hidden) then
      if(l1h>0) denominator=denominator+l1h
      if(l2h>0) denominator=denominator+l2h*h
    else
      if(l1w>0) denominator=denominator+l1w
      if(l2w>0) denominator=denominator+l2w*w
    end if
    where(denominator==0) denominator=eps32
  end subroutine
  subroutine update_w(self,x,w,h)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:),h(:,:)
    real(dp), intent(inout) :: w(:,:)
    real(dp), allocatable :: numerator(:,:),denominator(:,:),delta(:,:)
    call weights(self,x,w,h,numerator,denominator,.false.); delta=numerator/denominator
    if(self%gamma_/=1) delta=delta**self%gamma_
    w=w*delta
  end subroutine
  logical function small_motion(current,old,tol) result(small)
    real(dp), intent(in) :: current(:,:),old(:,:),tol
    real(dp) :: norm
    small=.false.
    if(tol<=0) return
    norm=sqrt(sum(current**2))
    if(norm==0) return ! source 0/0 is NaN, so it cannot trigger convergence.
    small=sqrt(sum((current-old)**2))/norm<=tol
  end function
  function solve_w(self,x,h,limit) result(w)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:),h(:,:)
    integer, intent(in) :: limit
    real(dp), allocatable :: w(:,:),old(:,:)
    integer :: i
    allocate(w(size(x,1),size(h,1))); w=sqrt(sum(x)/real(size(x),dp)/self%n_components_); old=w
    do i=1,limit
      call update_w(self,x,w,h)
      if(small_motion(w,old,self%tol)) exit
      old=w
    end do
  end function
  real(dp) function divergence(x,w,h,beta) result(value)
    real(dp), intent(in) :: x(:,:),w(:,:),h(:,:),beta
    real(dp), allocatable :: prediction(:,:)
    real(dp) :: target,estimated,ratio,total_power,mixed,total_x
    integer :: i,j
    prediction=matmul(w,h)
    if(beta==2) then
      value=.5_dp*sum((x-prediction)**2); return
    end if
    value=0; total_x=0; mixed=0
    if(beta==0) value=-real(size(x),dp)
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(x(i,j)<=eps32) cycle
        target=x(i,j); estimated=max(prediction(i,j),eps32); ratio=target/estimated
        if(beta==1) then
          value=value+target*log(ratio); total_x=total_x+target
        else if(beta==0) then
          value=value+ratio-log(ratio)
        else
          value=value+target**beta; mixed=mixed+target*estimated**(beta-1)
        end if
      end do
    end do
    if(beta==1) then
      value=value+dot_product(sum(w,dim=1),sum(h,dim=2))-total_x
    else if(beta/=0) then
      if(beta<0.and.any(prediction==0)) then
        value=ieee_value(0._dp,ieee_positive_inf); return
      end if
      total_power=sum(prediction**beta)
      value=(value-beta*mixed+(beta-1)*total_power)/(beta*(beta-1))
    end if
  end function
  real(dp) function step(self,x,w,h,restart) result(cost)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(inout) :: w(:,:),h(:,:)
    logical, intent(in) :: restart
    real(dp), allocatable :: numerator(:,:),denominator(:,:),power_h(:,:),working(:,:)
    real(dp) :: l1w,l2w,l1h,l2h
    if(restart) then
      working=solve_w(self,x,h,self%fresh_restarts_max_iter)
    else
      working=w; call update_w(self,x,working,h)
    end if
    if(self%beta_loss_<1) then
      where(working<epsilon(1._dp)) working=0
    end if
    if(.not.restart) w=working
    call regularization(self,size(x,1),size(x,2),l1w,l2w,l1h,l2h)
    cost=(divergence(x,working,h,self%beta_loss_)+l1w*sum(working)+l1h*sum(h)+ &
      l2w*sum(working**2)+l2h*sum(h**2))/size(x,1)
    call weights(self,x,working,h,numerator,denominator,.true.); power_h=h
    if(self%gamma_/=1) power_h=power_h**(1/self%gamma_)
    self%components_numerator_=self%rho_*self%components_numerator_+numerator*power_h
    self%components_denominator_=self%rho_*self%components_denominator_+denominator
    h=self%components_numerator_/self%components_denominator_
    if(self%gamma_/=1) h=h**self%gamma_
    if(self%beta_loss_<=1) then
      where(h<epsilon(1._dp)) h=0
    end if
  end function
  logical function converged(self,cost,h,old,n,batch,iteration) result(stopped)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp), intent(in) :: cost,h(:,:),old(:,:)
    integer, intent(in) :: n,batch
    integer(int64), intent(in) :: iteration
    real(dp) :: weight
    stopped=.false.; if(iteration==0) return
    if(ieee_is_nan(self%ewa_cost_)) then
      self%ewa_cost_=cost
    else
      weight=min(1._dp,real(batch,dp)/(real(n,dp)+1))
      self%ewa_cost_=(1-weight)*self%ewa_cost_+weight*cost
    end if
    if(small_motion(h,old,self%tol)) then
      stopped=.true.; return
    end if
    if(ieee_is_nan(self%ewa_cost_min_)) then
      self%ewa_cost_min_=self%ewa_cost_; self%no_improvement_=0
    else if(self%ewa_cost_<self%ewa_cost_min_) then
      self%ewa_cost_min_=self%ewa_cost_; self%no_improvement_=0
    else
      self%no_improvement_=self%no_improvement_+1
    end if
    if(self%max_no_improvement>=1) stopped=self%no_improvement_>=int(self%max_no_improvement,int64)
  end function
  function fit_transform(self,x,w,h) result(codes)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w(:,:),h(:,:)
    real(dp), allocatable :: codes(:,:),dictionary(:,:),old(:,:)
    real(dp) :: cost
    integer :: n,batches,first,last
    integer(int64) :: iteration,total
    call controls(self); call data_check(x,self%beta_loss); call resolve(self,x,h)
    call initialize(self,x,w,h,codes,dictionary); old=dictionary
    self%components_numerator_=dictionary; self%components_denominator_=0*dictionary+1
    self%ewa_cost_=ieee_value(0._dp,ieee_quiet_nan); self%ewa_cost_min_=self%ewa_cost_; self%no_improvement_=0
    n=size(x,1); batches=(n-1)/self%batch_size_+1; total=int(self%max_iter,int64)*batches; first=1; self%n_steps_=0
    do iteration=0_int64,total-1
      last=first+min(self%batch_size_,n-first+1)-1
      cost=step(self,x(first:last,:),codes(first:last,:),dictionary,self%fresh_restarts); self%n_steps_=iteration+1
      if(converged(self,cost,dictionary,old,n,last-first+1,iteration)) exit
      old=dictionary
      if(last==n) then
        first=1
      else
        first=last+1
      end if
    end do
    if(self%fresh_restarts) codes=solve_w(self,x,dictionary,self%transform_max_iter_)
    self%components_=dictionary; self%n_iter_=(self%n_steps_+batches-1)/batches
    self%reconstruction_err_=sqrt(2*max(0._dp,divergence(x,codes,dictionary,self%beta_loss_)))
  end function
  subroutine fit(self,x,w,h)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w(:,:),h(:,:)
    real(dp), allocatable :: codes(:,:)
    codes=self%fit_transform(x,w,h)
  end subroutine
  subroutine partial_fit(self,x,w,h)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w(:,:),h(:,:)
    real(dp), allocatable :: codes(:,:),dictionary(:,:)
    real(dp) :: cost
    call controls(self)
    if(.not.allocated(self%components_)) then
      call data_check(x,self%beta_loss); call resolve(self,x,h); call initialize(self,x,w,h,codes,dictionary)
      self%components_numerator_=dictionary; self%components_denominator_=0*dictionary+1
      self%n_steps_=0; self%n_iter_=0; self%reconstruction_err_=ieee_value(0._dp,ieee_quiet_nan)
      self%ewa_cost_=ieee_value(0._dp,ieee_quiet_nan); self%ewa_cost_min_=self%ewa_cost_; self%no_improvement_=0
    else
      call query(self,x); dictionary=self%components_
    end if
    ! Source partial_fit discards initial W and always solves a fresh coefficient batch.
    if(allocated(codes)) deallocate(codes)
    allocate(codes(0,0))
    call require(self%n_steps_<huge(self%n_steps_),'MiniBatchNMF step counter capacity exceeded')
    cost=step(self,x,codes,dictionary,.true.); self%components_=dictionary; self%n_steps_=self%n_steps_+1
  end subroutine
  function transform(self,x) result(codes)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: codes(:,:)
    call query(self,x); codes=solve_w(self,x,self%components_,self%transform_max_iter_)
  end function
  function inverse_transform(self,codes) result(x)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: codes(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'MiniBatchNMF is not fitted'); call check_x(codes)
    call require(size(codes,2)==self%n_components_,'MiniBatchNMF inverse component count mismatch')
    x=matmul(codes,self%components_)
  end function
end module
