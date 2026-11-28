! Dense sklearn 1.6.1 decomposition/_nmf.py MiniBatchNMF contracts (BSD-3-Clause).
! Upstream notices: LICENSES/scikit-learn-COPYING, LICENSES/numpy-* and THIRD_PARTY_NOTICES.md.
module sklearn_minibatch_nmf
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_sparse, only: csr_matrix
  use sklearn_nmf_sparse_data,only:nmf_sparse_cross
  use sklearn_stateless_preprocessing,only:sorted_sparse
  use,intrinsic::ieee_exceptions
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  private
  public :: mini_batch_nmf,nmf_initialize_factors
  real(dp), parameter :: eps32=1.1920928955078125e-7_dp
  interface nmf_initialize_factors
    module procedure nmf_initialize_dense,nmf_initialize_sparse
  end interface
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
    procedure,private::fit_dense=>fit,fit_csr=>fit_sparse
    generic::fit=>fit_dense,fit_csr
    procedure,private::fit_transform_dense=>fit_transform,fit_transform_csr=>fit_transform_sparse
    generic::fit_transform=>fit_transform_dense,fit_transform_csr
    procedure,private::partial_fit_dense=>partial_fit,partial_fit_csr=>partial_fit_sparse
    generic::partial_fit=>partial_fit_dense,partial_fit_csr
    procedure,private::transform_dense=>transform,transform_csr=>transform_sparse
    generic::transform=>transform_dense,transform_csr
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
  subroutine resolve(self,x,h,sparse)
    class(mini_batch_nmf), intent(inout) :: self
    real(dp),optional,intent(in)::x(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp), optional, intent(in) :: h(:,:)
    integer::n,p
    if(present(sparse))then
      n=sparse%n_rows;p=sparse%n_cols
    else
      n=size(x,1);p=size(x,2)
    end if
    self%n_components_=self%n_components
    if(self%n_components==0) then
      self%n_components_=p
      if(self%init=='custom'.and.present(h)) self%n_components_=size(h,1)
    end if
    call require(self%n_components_>=1.and.self%n_components_<=huge(1)-10,'MiniBatchNMF invalid component count')
    self%n_features_in_=p; self%batch_size_=min(self%batch_size,n); self%beta_loss_=self%beta_loss
    self%rho_=self%forget_factor**(real(self%batch_size_,dp)/n)
    self%gamma_=1
    if(self%beta_loss_<1) self%gamma_=1/(2-self%beta_loss_)
    if(self%beta_loss_>2) self%gamma_=1/(self%beta_loss_-1)
    self%transform_max_iter_=self%transform_max_iter
    if(self%transform_max_iter==0) self%transform_max_iter_=self%max_iter
  end subroutine
  subroutine initialize(self,x,w0,h0,w,h,sparse)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), optional, intent(in) :: x(:,:)
    type(csr_matrix), optional, intent(in) :: sparse
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    real(dp), allocatable, intent(out) :: w(:,:),h(:,:)
    real(dp), allocatable :: u(:,:),s(:),vt(:,:),xp(:),xn(:),yp(:),yn(:)
    real(dp) :: avg,scale,np,nm,sp,sn,mass
    type(legacy_random_state) :: rng
    integer :: n,p,k,i,j
    character(8) :: method
    if(present(sparse))then
      n=sparse%n_rows;p=sparse%n_cols
    else
      n=size(x,1);p=size(x,2)
    end if
    k=self%n_components_; method=self%init
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
    if(present(sparse))then
      avg=sum(sparse%data)/(real(n,dp)*p)
    else
      avg=sum(x)/real(size(x),dp)
    end if
    call rng%seed(self%random_state)
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
    if(present(sparse))then
      call randomized_svd(sparse,k,rng,s,u,vt)
    else
      call randomized_svd(x,k,rng,s,u,vt)
    end if
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
  function prepare_sparse(x,beta,mode,nonnegative)result(z)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::beta
    integer,intent(in)::mode ! 0 fit, 1 partial_fit, 2 transform
    logical,intent(in)::nonnegative
    type(csr_matrix)::z,canonical
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'MiniBatchNMF CSR shape must be nonempty')
    call require(all(ieee_is_finite(x%data)),'MiniBatchNMF CSR data must be finite')
    if(nonnegative)call require(all(x%data>=0),'MiniBatchNMF X must be nonnegative')
    if(mode==0)then
      ! Only fit calls X.min(), canonicalizing duplicates in Source.
      z=sorted_sparse(x,.true.)
      if(beta<=0)call require(int(size(z%data),int64)==int(z%n_rows,int64)*z%n_cols.and.all(z%data>0), &
        'MiniBatchNMF nonpositive beta requires strictly positive X')
    else
      z=x
    end if
    if(beta/=2)then
      canonical=sorted_sparse(z,.true.)
      call require(size(canonical%data)==size(z%data),'MiniBatchNMF sparse beta update rejects duplicate storage')
      ! Empty patterns align; a singleton zero also broadcasts in Source's
      ! transform-only update. Costs used by fit/partial_fit reject that case.
      if(.not.(mode==2.and.size(z%data)==1)) &
        call require(all(z%data/=0),'MiniBatchNMF sparse beta update rejects explicit zeros')
    end if
  end function
  function sparse_batch(x,first,last)result(z)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::first,last
    type(csr_matrix)::z
    integer::left,right
    left=x%indptr(first);right=x%indptr(last+1)-1
    z%n_rows=last-first+1;z%n_cols=x%n_cols
    z%indptr=x%indptr(first:last+1)-left+1;z%indices=x%indices(left:right);z%data=x%data(left:right)
  end function
  subroutine sparse_terms(x,w,h,beta,numerator,denominator,hidden)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::w(:,:),h(:,:),beta
    real(dp),allocatable,intent(out)::numerator(:,:),denominator(:,:)
    logical,intent(in)::hidden
    type(csr_matrix)::pattern
    real(dp),allocatable::row(:),column(:)
    real(dp)::estimate,value
    integer::n,p,k,i,j,e
    n=x%n_rows;p=x%n_cols;k=size(h,1)
    if(beta==2)then
      if(hidden)then
        numerator=transpose(nmf_sparse_cross(x,w,.true.))
        if(real(k,dp)*k*(real(n,dp)+p)<2._dp*n*k*p)then
          denominator=matmul(matmul(transpose(w),w),h)
        else
          allocate(denominator(k,p))
          do j=1,p
            column=matmul(w,h(:,j));denominator(:,j)=matmul(transpose(w),column)
          end do
        end if
      else
        numerator=nmf_sparse_cross(x,transpose(h),.false.);denominator=matmul(w,matmul(h,transpose(h)))
      end if
      return
    end if
    if(hidden)then
      allocate(numerator(k,p))
    else
      allocate(numerator(n,k))
    end if
    numerator=0;pattern=sorted_sparse(x,.true.)
    do i=1,n
      do e=x%indptr(i),x%indptr(i+1)-1
        ! Source _special_sparse_dot returns sorted CSR, but combines its data
        ! with the ORIGINAL unsorted X.data. Retain that observable ordering.
        j=pattern%indices(e);estimate=sum(w(i,:)*h(:,j))
        if(beta<2)estimate=max(estimate,eps32)
        if(beta==1)then
          value=x%data(e)/estimate
        else if(beta==0)then
          value=(1/estimate)**2*x%data(e)
        else
          value=estimate**(beta-2)*x%data(e)
        end if
        if(hidden)then
          numerator(:,j)=numerator(:,j)+w(i,:)*value
        else
          numerator(i,:)=numerator(i,:)+value*h(:,j)
        end if
      end do
    end do
    if(beta==1)then
      if(hidden)then
        column=sum(w,dim=1);where(column==0)column=1
        denominator=spread(column,2,p)
      else
        denominator=spread(sum(h,dim=2),1,n)
      end if
    else if(hidden)then
      allocate(denominator(k,p))
      do j=1,p
        column=matmul(w,h(:,j));if(beta<1)column=max(column,eps32)
        denominator(:,j)=matmul(transpose(w),column**(beta-1))
      end do
    else
      allocate(denominator(n,k))
      do i=1,n
        row=matmul(w(i,:),h);if(beta<1)row=max(row,eps32)
        denominator(i,:)=matmul(row**(beta-1),transpose(h))
      end do
    end if
  end subroutine
  real(dp)function sparse_divergence(x,w,h,beta)result(value)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::w(:,:),h(:,:),beta
    type(csr_matrix)::pattern
    real(dp),allocatable::cross(:,:),denominator(:,:),column(:)
    real(dp)::target,estimate,ratio,total_x,mixed,total_power
    integer::i,j,e
    if(beta==2)then
      call sparse_terms(x,w,h,beta,cross,denominator,.true.)
      value=.5_dp*(sum(x%data**2)+sum(denominator*h)-2*sum(nmf_sparse_cross(x,transpose(h),.false.)*w))
      return
    end if
    value=0;total_x=0;mixed=0;pattern=sorted_sparse(x,.true.)
    if(beta==0)value=-real(x%n_rows,dp)*x%n_cols
    do i=1,x%n_rows
      do e=x%indptr(i),x%indptr(i+1)-1
        target=x%data(e);if(target<=eps32)cycle
        j=pattern%indices(e);estimate=max(sum(w(i,:)*h(:,j)),eps32);ratio=target/estimate
        if(beta==1)then
          value=value+target*log(ratio);total_x=total_x+target
        else if(beta==0)then
          value=value+ratio-log(ratio)
        else
          value=value+target**beta;mixed=mixed+target*estimate**(beta-1)
        end if
      end do
    end do
    if(beta==1)then
      value=value+dot_product(sum(w,dim=1),sum(h,dim=2))-total_x
    else if(beta/=0)then
      total_power=0
      do j=1,x%n_cols
        column=matmul(w,h(:,j))
        if(beta<0.and.any(column==0))then
          value=ieee_value(0._dp,ieee_positive_inf);return
        end if
        total_power=total_power+sum(column**beta)
      end do
      value=(value-beta*mixed+(beta-1)*total_power)/(beta*(beta-1))
    end if
  end function
  function solve_w_sparse(self,x,h,limit)result(w)
    class(mini_batch_nmf),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::h(:,:)
    integer,intent(in)::limit
    real(dp),allocatable::w(:,:),old(:,:)
    integer::i
    allocate(w(x%n_rows,size(h,1)));w=sqrt(sum(x%data)/(real(x%n_rows,dp)*x%n_cols)/self%n_components_);old=w
    do i=1,limit
      call update_w(self,w=w,h=h,sparse=x)
      ! scipy.linalg.norm checks finite values even when tol == 0.
      call require(all(ieee_is_finite(w)).and.all(ieee_is_finite(old)),'MiniBatchNMF coefficient motion must be finite')
      if(small_motion(w,old,self%tol))exit
      old=w
    end do
  end function
  real(dp)function step_sparse(self,x,w,h,restart)result(cost)
    class(mini_batch_nmf),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(inout)::w(:,:),h(:,:)
    logical,intent(in)::restart
    real(dp),allocatable::working(:,:),numerator(:,:),denominator(:,:),power_h(:,:)
    real(dp)::l1w,l2w,l1h,l2h
    if(restart)then
      working=solve_w_sparse(self,x,h,self%fresh_restarts_max_iter)
    else
      working=w;call update_w(self,w=working,h=h,sparse=x)
    end if
    if(self%beta_loss_<1)where(working<epsilon(1._dp))working=0
    if(.not.restart)w=working
    call regularization(self,x%n_rows,x%n_cols,l1w,l2w,l1h,l2h)
    cost=(sparse_divergence(x,working,h,self%beta_loss_)+l1w*sum(working)+l1h*sum(h)+ &
      l2w*sum(working**2)+l2h*sum(h**2))/x%n_rows
    call weights(self,w=working,h=h,numerator=numerator,denominator=denominator,hidden=.true.,sparse=x)
    power_h=h;if(self%gamma_/=1)power_h=power_h**(1/self%gamma_)
    self%components_numerator_=self%rho_*self%components_numerator_+numerator*power_h
    self%components_denominator_=self%rho_*self%components_denominator_+denominator
    h=self%components_numerator_/self%components_denominator_
    if(self%gamma_/=1)h=h**self%gamma_
    if(self%beta_loss_<=1)where(h<epsilon(1._dp))h=0
  end function
  subroutine suspend_sparse_ieee(halting,raised)
    logical,intent(out)::halting(size(ieee_all)),raised(size(ieee_all))
    integer::i
    do i=1,size(ieee_all)
      call ieee_get_flag(ieee_all(i),raised(i));call ieee_get_halting_mode(ieee_all(i),halting(i))
      call ieee_set_halting_mode(ieee_all(i),.false.)
    end do
  end subroutine
  subroutine restore_sparse_ieee(halting,raised)
    logical,intent(in)::halting(size(ieee_all)),raised(size(ieee_all))
    integer::i
    do i=1,size(ieee_all)
      call ieee_set_flag(ieee_all(i),raised(i));call ieee_set_halting_mode(ieee_all(i),halting(i))
    end do
  end subroutine
  subroutine weights(self,x,w,h,numerator,denominator,hidden,sparse)
    class(mini_batch_nmf), intent(in) :: self
    real(dp),optional,intent(in)::x(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp), intent(in) :: w(:,:),h(:,:)
    real(dp), allocatable, intent(out) :: numerator(:,:),denominator(:,:)
    logical, intent(in) :: hidden
    real(dp), allocatable :: prediction(:,:),safe(:,:),power(:,:),ratio(:,:)
    real(dp) :: beta,l1w,l2w,l1h,l2h
    integer :: n,p,k
    if(present(sparse))then
      n=sparse%n_rows;p=sparse%n_cols
    else
      n=size(x,1);p=size(x,2)
    end if
    k=size(h,1); beta=self%beta_loss_
    call regularization(self,n,p,l1w,l2w,l1h,l2h)
    if(present(sparse))then
      call sparse_terms(sparse,w,h,beta,numerator,denominator,hidden)
    else if(beta==2) then
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
  subroutine update_w(self,x,w,h,sparse)
    class(mini_batch_nmf), intent(in) :: self
    real(dp),optional,intent(in)::x(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp), intent(in) :: h(:,:)
    real(dp), intent(inout) :: w(:,:)
    real(dp), allocatable :: numerator(:,:),denominator(:,:),delta(:,:)
    call weights(self,x,w,h,numerator,denominator,.false.,sparse); delta=numerator/denominator
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
  function fit_transform_sparse(self,x,w,h)result(codes)
    class(mini_batch_nmf),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::w(:,:),h(:,:)
    real(dp),allocatable::codes(:,:),dictionary(:,:),old(:,:)
    type(csr_matrix)::prepared,batch
    real(dp)::cost,value
    integer::n,batches,first,last
    integer(int64)::iteration,total
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call controls(self);prepared=prepare_sparse(x,self%beta_loss,0,.true.)
    call resolve(self,h=h,sparse=prepared)
    call initialize(self,w0=w,h0=h,w=codes,h=dictionary,sparse=prepared);old=dictionary
    self%components_numerator_=dictionary;self%components_denominator_=0*dictionary+1
    self%ewa_cost_=ieee_value(0._dp,ieee_quiet_nan);self%ewa_cost_min_=self%ewa_cost_;self%no_improvement_=0
    n=x%n_rows;batches=(n-1)/self%batch_size_+1;total=int(self%max_iter,int64)*batches;first=1;self%n_steps_=0
    call suspend_sparse_ieee(halting,raised)
    do iteration=0_int64,total-1
      last=first+min(self%batch_size_,n-first+1)-1;batch=sparse_batch(prepared,first,last)
      cost=step_sparse(self,batch,codes(first:last,:),dictionary,self%fresh_restarts);self%n_steps_=iteration+1
      if(iteration>0)call require(all(ieee_is_finite(dictionary)).and.all(ieee_is_finite(old)), &
        'MiniBatchNMF component motion must be finite')
      if(converged(self,cost,dictionary,old,n,last-first+1,iteration))exit
      old=dictionary
      if(last==n)then
        first=1
      else
        first=last+1
      end if
    end do
    if(self%fresh_restarts)codes=solve_w_sparse(self,prepared,dictionary,self%transform_max_iter_)
    self%components_=dictionary;self%n_iter_=(self%n_steps_+batches-1)/batches
    value=sparse_divergence(prepared,codes,dictionary,self%beta_loss_)
    if(self%beta_loss_/=2.and.value<0)value=0
    self%reconstruction_err_=sqrt(2*value)
    call restore_sparse_ieee(halting,raised)
  end function
  subroutine fit_sparse(self,x,w,h)
    class(mini_batch_nmf),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::w(:,:),h(:,:)
    real(dp),allocatable::codes(:,:)
    codes=self%fit_transform(x,w,h)
  end subroutine
  subroutine partial_fit_sparse(self,x,w,h)
    class(mini_batch_nmf),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::w(:,:),h(:,:)
    real(dp),allocatable::codes(:,:),dictionary(:,:)
    type(csr_matrix)::prepared
    real(dp)::cost
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call controls(self)
    if(.not.allocated(self%components_))then
      prepared=prepare_sparse(x,self%beta_loss,1,self%init/='custom')
      call resolve(self,h=h,sparse=prepared)
      call initialize(self,w0=w,h0=h,w=codes,h=dictionary,sparse=prepared)
      self%components_numerator_=dictionary;self%components_denominator_=0*dictionary+1
      self%n_steps_=0;self%n_iter_=0;self%reconstruction_err_=ieee_value(0._dp,ieee_quiet_nan)
      self%ewa_cost_=ieee_value(0._dp,ieee_quiet_nan);self%ewa_cost_min_=self%ewa_cost_;self%no_improvement_=0
    else
      call require(x%n_cols==self%n_features_in_,'MiniBatchNMF feature count mismatch')
      prepared=prepare_sparse(x,self%beta_loss_,1,.false.);dictionary=self%components_
    end if
    if(allocated(codes))deallocate(codes)
    allocate(codes(0,0))
    call require(self%n_steps_<huge(self%n_steps_),'MiniBatchNMF step counter capacity exceeded')
    call suspend_sparse_ieee(halting,raised)
    cost=step_sparse(self,prepared,codes,dictionary,.true.)
    self%components_=dictionary;self%n_steps_=self%n_steps_+1
    call restore_sparse_ieee(halting,raised)
  end subroutine
  function transform_sparse(self,x)result(codes)
    class(mini_batch_nmf),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::codes(:,:)
    type(csr_matrix)::prepared
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call require(allocated(self%components_),'MiniBatchNMF is not fitted')
    call require(x%n_cols==self%n_features_in_,'MiniBatchNMF feature count mismatch')
    prepared=prepare_sparse(x,self%beta_loss_,2,.false.)
    call suspend_sparse_ieee(halting,raised)
    codes=solve_w_sparse(self,prepared,self%components_,self%transform_max_iter_)
    call restore_sparse_ieee(halting,raised)
  end function
  function inverse_transform(self,codes) result(x)
    class(mini_batch_nmf), intent(in) :: self
    real(dp), intent(in) :: codes(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'MiniBatchNMF is not fitted'); call check_x(codes)
    call require(size(codes,2)==self%n_components_,'MiniBatchNMF inverse component count mismatch')
    x=matmul(codes,self%components_)
  end function
  ! Shared internal initialization adapter; not re-exported through sklearn.
  subroutine nmf_initialize_dense(x,k,method,seed,w,h,w0,h0)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: k,seed
    character(*), intent(in) :: method
    real(dp), allocatable, intent(out) :: w(:,:),h(:,:)
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    type(mini_batch_nmf) :: initializer
    initializer%n_components_=k;initializer%init=method;initializer%random_state=seed
    call initialize(initializer,x,w0,h0,w,h)
  end subroutine
  subroutine nmf_initialize_sparse(x,k,method,seed,w,h,w0,h0)
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: k,seed
    character(*), intent(in) :: method
    real(dp), allocatable, intent(out) :: w(:,:),h(:,:)
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    type(mini_batch_nmf) :: initializer
    initializer%n_components_=k;initializer%init=method;initializer%random_state=seed
    call initialize(initializer,w0=w0,h0=h0,w=w,h=h,sparse=x)
  end subroutine
end module
