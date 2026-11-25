! Dense covariance estimators. Shrinkage formulas follow sklearn/covariance/
! _shrunk_covariance.py (scikit-learn 1.6.1, BSD-3-Clause; see THIRD_PARTY_NOTICES).
module sklearn_covariance
  use sklearn_covariance_functions,only:covariance_estimate,empirical_array,empirical_vector,ledoit_array,oas_array, &
    shrink_array_2,shrink_array_3,shrink_array_4,shrink_array_5, &
    shrink_array_6,shrink_array_7,shrink_array_8,shrink_array_9, &
    shrink_array_10,shrink_array_11,shrink_array_12,shrink_array_13, &
    shrink_array_14,shrink_array_15
  use sklearn_shrinkage,only:ledoit_wolf_shrinkage
  use sklearn_least_angle,only:lars_path_gram,lars_path_result
  use sklearn_resampling, only: index_split
  use sklearn_robust_covariance, only: min_cov_det,elliptic_envelope,fast_mcd
  use sklearn_common
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf,ieee_positive_inf,ieee_quiet_nan
  use, intrinsic :: ieee_exceptions, only: ieee_invalid,ieee_divide_by_zero,ieee_overflow, &
    ieee_get_halting_mode,ieee_set_halting_mode,ieee_get_flag,ieee_set_flag
  implicit none
  private
  public :: min_cov_det,elliptic_envelope,fast_mcd
  public :: ledoit_wolf_shrinkage
  public :: empirical_covariance,shrunk_covariance,ledoit_wolf,oas,graphical_lasso
  public :: graphical_lasso_cv
  type :: empirical_covariance
    logical :: assume_centered=.false.
    real(dp), allocatable :: location_(:),covariance_(:,:),precision_(:,:)
  contains
    procedure :: fit=>empirical_fit
    procedure :: mahalanobis=>cov_mahalanobis
    procedure :: score=>cov_score
    procedure :: error_norm=>cov_error
  end type
  type, extends(empirical_covariance) :: shrunk_covariance
    real(dp) :: shrinkage=0.1_dp
  contains
    procedure :: fit=>shrunk_fit
  end type
  type, extends(empirical_covariance) :: ledoit_wolf
    integer :: block_size=1000
    real(dp) :: shrinkage_=0
  contains
    procedure :: fit=>ledoit_fit
  end type
  type, extends(empirical_covariance) :: oas
    real(dp) :: shrinkage_=0
  contains
    procedure :: fit=>oas_fit
  end type
  type, extends(empirical_covariance) :: graphical_lasso
    real(dp) :: alpha=0.01_dp,tol=1.e-4_dp,enet_tol=1.e-4_dp,eps=epsilon(1._dp)
    real(dp) :: objective_=0,dual_gap_=0
    integer :: max_iter=100,n_iter_=0
    character(16) :: covariance='empirical',mode='cd'
    logical :: converged_=.false.
    real(dp), allocatable :: costs_(:,:)
  contains
    procedure :: fit=>graphical_fit
  end type
  type :: graphical_cv_results
    real(dp), allocatable :: alphas(:),split_test_score(:,:),mean_test_score(:),std_test_score(:)
    real(dp), allocatable :: empirical_covariance(:,:,:),empirical_precision(:,:,:),empirical_test_covariance(:,:,:)
  end type
  type, extends(graphical_lasso) :: graphical_lasso_cv
    integer :: n_alphas=4,n_refinements=4,cv=5,n_features_in_=0
    real(dp) :: alpha_=0,best_score_=0
    real(dp), allocatable :: alphas(:)
    type(index_split), allocatable :: splits(:),cv_splits_(:)
    type(graphical_cv_results) :: cv_results_
  contains
    procedure :: fit=>graphical_cv_fit
    procedure :: score=>graphical_cv_score
  end type
  public::covariance_estimate
  interface empirical_covariance
    module procedure empirical_array,empirical_vector
  end interface
  interface ledoit_wolf
    module procedure ledoit_array
  end interface
  interface oas
    module procedure oas_array
  end interface
  interface shrunk_covariance
    module procedure shrink_array_2
    module procedure shrink_array_3
    module procedure shrink_array_4
    module procedure shrink_array_5
    module procedure shrink_array_6
    module procedure shrink_array_7
    module procedure shrink_array_8
    module procedure shrink_array_9
    module procedure shrink_array_10
    module procedure shrink_array_11
    module procedure shrink_array_12
    module procedure shrink_array_13
    module procedure shrink_array_14
    module procedure shrink_array_15
  end interface
contains
  ! sklearn 1.6.1 covariance/_graph_lasso.py and linear_model/_cd_fast.pyx.
  ! Cyclic Gram coordinate descent, including the upstream inner dual-gap test.
  subroutine graphical_cd(gram,row,alpha,coefs,max_iter,tol)
    real(dp), intent(in) :: gram(:,:),row(:),alpha,tol
    real(dp), intent(inout) :: coefs(:)
    integer, intent(in) :: max_iter
    real(dp) :: h(size(row)),old,value,change,largest,dual_norm,constant,gap,residual2,row_norm2,product
    integer :: it,j
    logical :: check_gap
    h=matmul(gram,coefs); row_norm2=sum(row**2)
    do it=1,max_iter
      change=0; largest=0
      do j=1,size(row)
        if(gram(j,j)==0) cycle
        old=coefs(j); h=h-old*gram(:,j); value=row(j)-h(j)
        coefs(j)=sign(max(abs(value)-alpha,0._dp),value)/gram(j,j)
        h=h+coefs(j)*gram(:,j)
        change=max(change,abs(coefs(j)-old)); largest=max(largest,abs(coefs(j)))
      end do
      check_gap=largest==0.or.it==max_iter
      if(largest>0) check_gap=check_gap.or.change/largest<tol
      if(.not.check_gap) cycle
      product=dot_product(row,coefs); dual_norm=maxval(abs(row-h))
      residual2=row_norm2+dot_product(coefs,h)-2*product
      constant=1
      if(dual_norm>alpha) constant=alpha/dual_norm
      gap=0.5_dp*residual2*(1+constant**2)+alpha*sum(abs(coefs))-constant*row_norm2+constant*product
      if(gap<tol*row_norm2) exit
      ! A zero right-hand side has an exact zero solution and zero gap.
      if(row_norm2==0.and.all(coefs==0)) exit
    end do
  end subroutine

  function graphical_inverse(a) result(inverse)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inverse(size(a,1),size(a,2)),l(size(a,1),size(a,2)),unit(size(a,1))
    integer :: j
    l=cholesky_factor(a)
    do j=1,size(a,1)
      unit=0; unit(j)=1; inverse(:,j)=solve_upper(transpose(l),solve_lower(l,unit))
    end do
    inverse=0.5_dp*(inverse+transpose(inverse))
  end function

  subroutine graphical_diagnostics(emp_cov,precision,alpha,cost,gap)
    real(dp), intent(in) :: emp_cov(:,:),precision(:,:),alpha
    real(dp), intent(out) :: cost,gap
    real(dp) :: l(size(precision,1),size(precision,2)),penalty,trace_product,logdet
    integer :: j,p
    p=size(precision,1); l=cholesky_factor(precision); penalty=0; logdet=0
    do j=1,p
      penalty=penalty+sum(abs(precision(:j-1,j)))+sum(abs(precision(j+1:,j)))
      logdet=logdet+2*log(l(j,j))
    end do
    trace_product=sum(emp_cov*precision)
    gap=trace_product-p+alpha*penalty
    ! Match sklearn's shifted objective, including its Gaussian constant.
    cost=-logdet+trace_product+2*p*log(2*acos(-1._dp))+alpha*penalty
  end subroutine

  subroutine graphical_fit(self,x)
    class(graphical_lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: empirical(:,:),centered(:,:),gram(:,:),coefs(:),row(:),history(:,:)
    type(lars_path_result)::lar_path
    real(dp) :: denominator
    real(dp) :: values(size(x,2)),vectors(size(x,2),size(x,2))
    integer :: p,n,i,j,it,indices(size(x,2)-1)
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(n>=2.and.p>=2,'GraphicalLasso requires at least two samples and features')
    call require(all(ieee_is_finite([self%alpha,self%tol,self%enet_tol,self%eps])), &
      'nonfinite GraphicalLasso controls')
    call require(self%alpha>=0,'GraphicalLasso alpha must be nonnegative')
    call require(self%tol>0.and.self%enet_tol>0.and.self%eps>0,'invalid GraphicalLasso tolerances')
    call require(self%max_iter>0,'GraphicalLasso max_iter must be positive')
    call require(self%mode=='cd'.or.self%mode=='lars','invalid GraphicalLasso solver mode')
    call require(self%covariance=='empirical'.or.self%covariance=='precomputed', &
      'invalid GraphicalLasso covariance mode')
    if(self%covariance=='precomputed') then
      call require(n==p,'GraphicalLasso precomputed covariance must be square')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x),abs(transpose(x)))), &
        'GraphicalLasso precomputed covariance must be symmetric')
      empirical=0.5_dp*(x+transpose(x)); self%location_=spread(0._dp,1,p)
      call symmetric_eigen(empirical,values,vectors)
      call require(minval(values)>= -1.e-10_dp*maxval(abs(values)), &
        'GraphicalLasso covariance must be positive semidefinite')
    else
      self%location_=sum(x,dim=1)/n
      if(self%assume_centered) self%location_=0
      centered=x-spread(self%location_,1,n); empirical=matmul(transpose(centered),centered)/n
    end if
    do j=1,p
      call require(empirical(j,j)>0,'GraphicalLasso requires positive feature variances')
    end do
    self%n_iter_=0; self%converged_=.false.
    if(self%alpha==0) then
      self%covariance_=empirical; self%precision_=graphical_inverse(empirical)
      call graphical_diagnostics(empirical,self%precision_,self%alpha,self%objective_,self%dual_gap_)
      if(allocated(self%costs_)) deallocate(self%costs_)
      allocate(self%costs_(1,2)); self%costs_(1,:)=[self%objective_,self%dual_gap_]
      self%converged_=.true.; return
    end if
    self%covariance_=0.95_dp*empirical
    do j=1,p
      self%covariance_(j,j)=empirical(j,j)
    end do
    self%precision_=graphical_inverse(self%covariance_); allocate(history(self%max_iter,2))
    do it=1,self%max_iter
      do j=1,p
        indices=pack([(i,i=1,p)],[(i/=j,i=1,p)])
        gram=self%covariance_(indices,indices); row=empirical(indices,j)
        if(self%mode=='cd')then
          coefs=-self%precision_(indices,j)/(self%precision_(j,j)+1000*self%eps)
          call graphical_cd(gram,row,self%alpha,coefs,self%max_iter,self%enet_tol)
        else
          lar_path=lars_path_gram(row,gram,p-1,alpha_min=self%alpha/(p-1),eps=self%eps,method='lar')
          coefs=lar_path%coefficients(:,size(lar_path%alphas))
        end if
        denominator=self%covariance_(j,j)-dot_product(self%covariance_(indices,j),coefs)
        call require(denominator>0,'GraphicalLasso nonpositive Schur complement; ill-conditioned fit')
        self%precision_(j,j)=1/denominator
        self%precision_(indices,j)=-self%precision_(j,j)*coefs
        self%precision_(j,indices)=self%precision_(indices,j)
        self%covariance_(indices,j)=matmul(gram,coefs)
        self%covariance_(j,indices)=self%covariance_(indices,j)
      end do
      call require(all(ieee_is_finite(self%precision_)),'GraphicalLasso nonfinite precision')
      call graphical_diagnostics(empirical,self%precision_,self%alpha,self%objective_,self%dual_gap_)
      history(it,:)=[self%objective_,self%dual_gap_]
      if(abs(self%dual_gap_)<self%tol) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter); self%costs_=history(:self%n_iter_,:)
  end subroutine

  ! GraphicalLassoCV and its private warm-start path follow sklearn 1.6.1
  ! covariance/_graph_lasso.py. Failed candidates are data, not fatal fits.
  subroutine graph_cv_fp_begin(modes,flags)
    logical, intent(out) :: modes(3),flags(3)
    call ieee_get_halting_mode(ieee_invalid,modes(1)); call ieee_get_flag(ieee_invalid,flags(1))
    call ieee_get_halting_mode(ieee_divide_by_zero,modes(2)); call ieee_get_flag(ieee_divide_by_zero,flags(2))
    call ieee_get_halting_mode(ieee_overflow,modes(3)); call ieee_get_flag(ieee_overflow,flags(3))
    call ieee_set_halting_mode(ieee_invalid,.false.)
    call ieee_set_halting_mode(ieee_divide_by_zero,.false.)
    call ieee_set_halting_mode(ieee_overflow,.false.)
  end subroutine
  subroutine graph_cv_fp_end(modes,flags)
    logical, intent(in) :: modes(3),flags(3)
    call ieee_set_flag(ieee_invalid,flags(1)); call ieee_set_flag(ieee_divide_by_zero,flags(2))
    call ieee_set_flag(ieee_overflow,flags(3))
    call ieee_set_halting_mode(ieee_invalid,modes(1))
    call ieee_set_halting_mode(ieee_divide_by_zero,modes(2))
    call ieee_set_halting_mode(ieee_overflow,modes(3))
  end subroutine

  function graph_cv_cov(x,centered) result(cov)
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: centered
    real(dp) :: cov(size(x,2),size(x,2)),work(size(x,1),size(x,2)),mean(size(x,2))
    mean=0; if(.not.centered) mean=sum(x,dim=1)/size(x,1)
    work=x-spread(mean,1,size(x,1)); cov=matmul(transpose(work),work)/size(x,1)
  end function

  function graph_cv_pinv(cov) result(precision)
    real(dp), intent(in) :: cov(:,:)
    real(dp) :: precision(size(cov,1),size(cov,1)),values(size(cov,1)),vectors(size(cov,1),size(cov,1))
    real(dp) :: inverse(size(cov,1)),cutoff
    integer :: i
    call symmetric_eigen(cov,values,vectors)
    cutoff=size(cov,1)*epsilon(1._dp)*maxval(abs(values)); inverse=0
    do i=1,size(values)
      if(abs(values(i))>cutoff) inverse(i)=1/values(i)
    end do
    precision=matmul(vectors*spread(inverse,1,size(values)),transpose(vectors))
  end function

  real(dp) function graph_cv_logdet(precision) result(value)
    real(dp), intent(in) :: precision(:,:)
    real(dp) :: work(size(precision,1),size(precision,2)),row(size(precision,1)),pivot
    integer :: i,j,k,sign_det,p
    value=ieee_value(0._dp,ieee_negative_inf); p=size(precision,1)
    if(.not.all(ieee_is_finite(precision))) return
    work=precision; sign_det=1; value=0
    do i=1,p
      k=i-1+maxloc(abs(work(i:,i)),dim=1)
      if(k/=i) then
        row=work(i,:); work(i,:)=work(k,:); work(k,:)=row; sign_det=-sign_det
      end if
      pivot=work(i,i)
      if(pivot==0.or..not.ieee_is_finite(pivot)) then
        value=ieee_value(0._dp,ieee_negative_inf); return
      end if
      if(pivot<0) sign_det=-sign_det
      value=value+log(abs(pivot))
      do j=i+1,p
        work(j,i+1:)=work(j,i+1:)-(work(j,i)/pivot)*work(i,i+1:)
      end do
    end do
    if(sign_det<0) value=ieee_value(0._dp,ieee_negative_inf)
  end function

  real(dp) function graph_cv_likelihood(cov,precision) result(value)
    real(dp), intent(in) :: cov(:,:),precision(:,:)
    value=(graph_cv_logdet(precision)-sum(cov*precision)-size(cov,1)*log(2*acos(-1._dp)))/2
  end function

  function graph_cv_inverse(cov,ok) result(precision)
    real(dp), intent(in) :: cov(:,:)
    logical, intent(out) :: ok
    real(dp) :: precision(size(cov,1),size(cov,1)),work(size(cov,1),size(cov,1)),row(size(cov,1)),pivot
    integer :: i,j,k,p
    p=size(cov,1); work=cov; precision=0; ok=.false.
    do i=1,p
      precision(i,i)=1
    end do
    do i=1,p
      k=i-1+maxloc(abs(work(i:,i)),dim=1)
      if(k/=i) then
        row=work(i,:); work(i,:)=work(k,:); work(k,:)=row
        row=precision(i,:); precision(i,:)=precision(k,:); precision(k,:)=row
      end if
      pivot=work(i,i); if(pivot==0.or..not.ieee_is_finite(pivot)) return
      work(i,:)=work(i,:)/pivot; precision(i,:)=precision(i,:)/pivot
      do j=1,p
        if(j==i) cycle
        pivot=work(j,i); work(j,:)=work(j,:)-pivot*work(i,:)
        precision(j,:)=precision(j,:)-pivot*precision(i,:)
      end do
    end do
    ok=all(ieee_is_finite(precision))
  end function

  subroutine graph_cv_solve(self,emp_cov,alpha,initial,max_iter,cov,precision,costs,n_iter,converged,ok)
    class(graphical_lasso_cv), intent(in) :: self
    real(dp), intent(in) :: emp_cov(:,:),alpha,initial(:,:)
    integer, intent(in) :: max_iter
    real(dp), allocatable, intent(out) :: cov(:,:),precision(:,:),costs(:,:)
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged,ok
    real(dp), allocatable :: gram(:,:),row(:),coefs(:),history(:,:)
    type(lars_path_result)::lar_path
    logical::lar_ok
    real(dp) :: denominator,gap,cost,penalty
    integer :: i,j,p,it,indices(size(emp_cov,1)-1)
    p=size(emp_cov,1); n_iter=1; converged=.false.; ok=.false.
    if(alpha==0) then
      cov=emp_cov; precision=graph_cv_inverse(emp_cov,ok); if(.not.ok) return
      gap=sum(emp_cov*precision)-p
      cost=-graph_cv_logdet(precision)+sum(emp_cov*precision)+2*p*log(2*acos(-1._dp))
      allocate(costs(1,2)); costs(1,:)=[cost,gap]; n_iter=0; converged=abs(gap)<self%tol; return
    end if
    cov=.95_dp*initial
    do j=1,p
      cov(j,j)=emp_cov(j,j)
    end do
    if(.not.all(ieee_is_finite(cov))) return
    precision=graph_cv_pinv(cov); allocate(history(max_iter,2))
    do it=1,max_iter
      do j=1,p
        indices=pack([(i,i=1,p)],[(i/=j,i=1,p)])
        gram=cov(indices,indices); row=emp_cov(indices,j)
        if(self%mode=='cd')then
          coefs=-precision(indices,j)/(precision(j,j)+1000*self%eps)
          call graphical_cd(gram,row,alpha,coefs,max_iter,self%enet_tol)
        else
          ! Automatic grids for exactly diagonal covariance can contain NaNs.
          ! Treat these as failed CV candidates, as in the CD path.
          if(.not.ieee_is_finite(alpha))return
          if(.not.all(ieee_is_finite(gram)).or..not.all(ieee_is_finite(row)))return
          lar_path=lars_path_gram(row,gram,p-1,alpha_min=alpha/(p-1),eps=self%eps,method='lar',success=lar_ok)
          if(.not.lar_ok)return
          coefs=lar_path%coefficients(:,size(lar_path%alphas))
        end if
        if(.not.all(ieee_is_finite(coefs))) return
        denominator=cov(j,j)-dot_product(cov(indices,j),coefs)
        if(denominator==0) return
        precision(j,j)=1/denominator
        precision(indices,j)=-precision(j,j)*coefs; precision(j,indices)=precision(indices,j)
        cov(indices,j)=matmul(gram,coefs); cov(j,indices)=cov(indices,j)
      end do
      if(.not.ieee_is_finite(sum(precision))) return
      penalty=0
      do j=1,p
        penalty=penalty+sum(abs(precision(:j-1,j)))+sum(abs(precision(j+1:,j)))
      end do
      gap=sum(emp_cov*precision)-p+alpha*penalty
      cost=-graph_cv_logdet(precision)+sum(emp_cov*precision)+2*p*log(2*acos(-1._dp))+alpha*penalty
      history(it,:)=[cost,gap]
      if(abs(gap)<self%tol) then
        converged=.true.; exit
      end if
      if(.not.ieee_is_finite(cost).and.it>1) return
    end do
    n_iter=min(it,max_iter); if(max_iter==0) n_iter=1
    costs=history(:min(it,max_iter),:); ok=.true.
  end subroutine

  subroutine graph_cv_folds(self,n)
    class(graphical_lasso_cv), intent(inout) :: self
    integer, intent(in) :: n
    integer :: f,i,width,first
    if(allocated(self%splits)) then
      self%cv_splits_=self%splits
      call require(size(self%cv_splits_)>0,'GraphicalLassoCV needs at least one split')
    else
      call require(self%cv>=2.and.self%cv<=n,'invalid GraphicalLassoCV cv count')
      if(allocated(self%cv_splits_)) deallocate(self%cv_splits_)
      allocate(self%cv_splits_(self%cv)); first=1
      do f=1,self%cv
        width=n/self%cv; if(f<=mod(n,self%cv)) width=width+1
        self%cv_splits_(f)%test=[(i,i=first,first+width-1)]
        self%cv_splits_(f)%train=pack([(i,i=1,n)],[(i<first.or.i>=first+width,i=1,n)])
        first=first+width
      end do
    end if
    do f=1,size(self%cv_splits_)
      call require(allocated(self%cv_splits_(f)%train).and.allocated(self%cv_splits_(f)%test), &
        'GraphicalLassoCV split indices are unallocated')
      call require(size(self%cv_splits_(f)%train)>0.and.size(self%cv_splits_(f)%test)>0, &
        'GraphicalLassoCV split indices must be nonempty')
      call require(all(self%cv_splits_(f)%train>=1).and.all(self%cv_splits_(f)%train<=n).and. &
        all(self%cv_splits_(f)%test>=1).and.all(self%cv_splits_(f)%test<=n), &
        'GraphicalLassoCV split index is out of bounds')
    end do
  end subroutine

  function graph_cv_logspace(high,low,n) result(grid)
    real(dp), intent(in) :: high,low
    integer, intent(in) :: n
    real(dp) :: grid(n),lo,hi
    integer :: i
    if(.not.all(ieee_is_finite([high,low])).or.high<=0.or.low<=0) then
      grid=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    lo=log10(low); hi=log10(high)
    do i=1,n
      grid(i)=10._dp**(hi+(lo-hi)*real(i-1,dp)/real(n-1,dp))
    end do
  end function

  function graph_cv_order(path) result(order)
    real(dp), intent(in) :: path(:)
    integer :: order(size(path)),i,j,v
    order=[(i,i=1,size(path))]
    do i=2,size(path)
      v=order(i); j=i-1
      do while(j>=1)
        ! Unlike argsort(-path), this preserves Python's stable NaN ordering.
        if(.not.(path(order(j))<path(v))) exit
        order(j+1)=order(j); j=j-1
      end do
      order(j+1)=v
    end do
  end function

  subroutine graphical_cv_fit(self,x)
    class(graphical_lasso_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: empirical(:,:),grid(:),path(:),scores(:,:),new_scores(:,:),cov(:,:),precision(:,:), &
      initial(:,:),train_cov(:,:),test_cov(:,:),costs(:,:),previous(:),values(:),centered(:,:)
    integer, allocatable :: train(:),test(:),order(:)
    real(dp) :: high,low,mean,best,negative_inf
    integer :: p,n,refinements,ref,f,j,i,best_index,last_finite,nold,nfolds,it,npth
    logical :: modes(3),flags(3),ok,converged
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(p>=2,'GraphicalLassoCV requires at least two features')
    call require(maxval(abs(x))<=sqrt(huge(1._dp))/(16*real(n,dp)*real(p,dp)), &
      'GraphicalLassoCV data exceeds safe covariance range')
    call require(all(ieee_is_finite([self%tol,self%enet_tol,self%eps])),'nonfinite GraphicalLassoCV controls')
    call require(self%tol>0.and.self%enet_tol>0.and.self%eps>0,'invalid GraphicalLassoCV tolerances')
    call require(self%max_iter>=0,'GraphicalLassoCV max_iter must be nonnegative')
    call require(self%mode=='cd'.or.self%mode=='lars','invalid GraphicalLassoCV solver mode')
    call require(self%covariance=='empirical','GraphicalLassoCV requires empirical covariance mode')
    call require(self%n_refinements>=1,'GraphicalLassoCV n_refinements must be positive')
    refinements=self%n_refinements
    empirical=graph_cv_cov(x,self%assume_centered)
    if(allocated(self%alphas)) then
      call require(size(self%alphas)>=2,'GraphicalLassoCV needs at least two explicit alphas')
      call require(all(ieee_is_finite(self%alphas)),'GraphicalLassoCV alphas must be finite')
      call require(all(self%alphas>0),'GraphicalLassoCV explicit alphas must be positive')
      grid=self%alphas; refinements=1
    else
      call require(self%n_alphas>=2,'GraphicalLassoCV n_alphas must be at least two')
      high=0
      do j=1,p
        high=max(high,maxval(abs(empirical(:j-1,j))),maxval(abs(empirical(j+1:,j))))
      end do
      if(.01_dp*high==0) then
        allocate(grid(self%n_alphas)); grid=ieee_value(0._dp,ieee_quiet_nan); grid(1)=high
      else
        grid=graph_cv_logspace(high,.01_dp*high,self%n_alphas)
      end if
    end if
    call graph_cv_folds(self,n); nfolds=size(self%cv_splits_)
    self%n_features_in_=p; self%location_=sum(x,dim=1)/n
    if(self%assume_centered) self%location_=0
    negative_inf=ieee_value(0._dp,ieee_negative_inf)
    allocate(path(0),scores(0,nfolds)); call graph_cv_fp_begin(modes,flags)
    do ref=1,refinements
      allocate(new_scores(size(grid),nfolds))
      do f=1,nfolds
        train=self%cv_splits_(f)%train; test=self%cv_splits_(f)%test
        ! Source path always centers both parts, even when the final fit does not.
        train_cov=graph_cv_cov(x(train,:),.false.); test_cov=graph_cv_cov(x(test,:),.false.)
        initial=train_cov
        do j=1,size(grid)
          call graph_cv_solve(self,train_cov,grid(j),initial,int(.1_dp*self%max_iter), &
            cov,precision,costs,it,converged,ok)
          if(grid(j)==0.and..not.ok) then
            call graph_cv_fp_end(modes,flags)
            call require(.false.,'GraphicalLassoCV zero-alpha covariance is singular')
          end if
          mean=negative_inf
          if(ok) then
            initial=cov; mean=graph_cv_likelihood(test_cov,precision)
          end if
          if(.not.ieee_is_finite(mean)) mean=negative_inf
          new_scores(j,f)=mean
        end do
      end do
      nold=size(path); path=[path,grid]
      block
        real(dp) :: combined(size(path),nfolds)
        combined(:nold,:)=scores; combined(nold+1:,:)=new_scores; scores=combined
      end block
      order=graph_cv_order(path); path=path(order); scores=scores(order,:)
      best=negative_inf; best_index=1; last_finite=1
      do j=1,size(path)
        mean=sum(scores(j,:))/nfolds
        if(mean>=.1_dp/epsilon(1._dp)) mean=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(mean)) last_finite=j
        if(mean>=best) then
          best=mean; best_index=j
        end if
      end do
      if(best_index==1) then
        high=path(1); low=path(2)
      else if(best_index==last_finite.and.best_index/=size(path)) then
        high=path(best_index); low=path(best_index+1)
      else if(best_index==size(path)) then
        high=path(best_index); low=.01_dp*high
      else
        high=path(best_index-1); low=path(best_index+1)
      end if
      if(.not.allocated(self%alphas)) then
        previous=graph_cv_logspace(high,low,self%n_alphas+2); grid=previous(2:size(previous)-1)
      end if
      deallocate(new_scores)
    end do
    self%alpha_=path(best_index); self%alpha=self%alpha_; self%best_score_=best
    npth=size(path); self%cv_results_%alphas=[path,0._dp]
    if(allocated(self%cv_results_%split_test_score)) deallocate(self%cv_results_%split_test_score)
    allocate(self%cv_results_%split_test_score(npth+1,nfolds)); self%cv_results_%split_test_score(:npth,:)=scores
    if(allocated(self%cv_results_%empirical_covariance)) deallocate(self%cv_results_%empirical_covariance, &
      self%cv_results_%empirical_precision,self%cv_results_%empirical_test_covariance)
    allocate(self%cv_results_%empirical_covariance(p,p,nfolds),self%cv_results_%empirical_precision(p,p,nfolds), &
      self%cv_results_%empirical_test_covariance(p,p,nfolds))
    do f=1,nfolds
      train=self%cv_splits_(f)%train; test=self%cv_splits_(f)%test
      train_cov=graph_cv_cov(x(train,:),.false.); precision=graph_cv_pinv(train_cov)
      centered=x(test,:)-spread(sum(x(train,:),dim=1)/size(train),1,size(test))
      test_cov=graph_cv_cov(centered,.true.)
      self%cv_results_%empirical_covariance(:,:,f)=train_cov
      self%cv_results_%empirical_precision(:,:,f)=precision
      self%cv_results_%empirical_test_covariance(:,:,f)=test_cov
      self%cv_results_%split_test_score(npth+1,f)=graph_cv_likelihood(test_cov,precision)
    end do
    allocate(values(npth+1))
    do j=1,npth+1
      values(j)=sum(self%cv_results_%split_test_score(j,:))/nfolds
    end do
    self%cv_results_%mean_test_score=values
    do j=1,npth+1
      mean=values(j); values(j)=ieee_value(0._dp,ieee_quiet_nan)
      if(ieee_is_finite(mean)) values(j)=sqrt(sum((self%cv_results_%split_test_score(j,:)-mean)**2)/nfolds)
    end do
    self%cv_results_%std_test_score=values
    call graph_cv_solve(self,empirical,self%alpha_,empirical,self%max_iter, &
      cov,precision,costs,it,converged,ok)
    call graph_cv_fp_end(modes,flags)
    call require(ok,'GraphicalLassoCV final fit is ill-conditioned')
    self%covariance_=cov; self%precision_=precision; self%costs_=costs; self%n_iter_=it; self%converged_=converged
    self%objective_=ieee_value(0._dp,ieee_positive_inf); self%dual_gap_=self%objective_
    if(size(costs,1)>0) then
      self%objective_=costs(size(costs,1),1); self%dual_gap_=costs(size(costs,1),2)
    end if
  end subroutine

  real(dp) function graphical_cv_score(self,x) result(value)
    class(graphical_lasso_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: centered(size(x,1),size(x,2))
    logical :: modes(3),flags(3)
    call require(allocated(self%precision_),'GraphicalLassoCV is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'GraphicalLassoCV query feature count mismatch')
    call graph_cv_fp_begin(modes,flags)
    centered=x-spread(self%location_,1,size(x,1))
    value=graph_cv_likelihood(graph_cv_cov(centered,.true.),self%precision_)
    call graph_cv_fp_end(modes,flags)
  end function

  subroutine empirical_fit(self,x)
    class(empirical_covariance), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: centered(size(x,1),size(x,2))
    call check_x(x); self%location_=sum(x,dim=1)/size(x,1)
    if(self%assume_centered) self%location_=0
    centered=x-spread(self%location_,1,size(x,1))
    self%covariance_=matmul(transpose(centered),centered)/size(x,1)
    call set_precision(self)
  end subroutine
  subroutine set_precision(self)
    class(empirical_covariance), intent(inout) :: self
    real(dp) :: values(size(self%covariance_,1)),vectors(size(self%covariance_,1),size(self%covariance_,1))
    real(dp) :: inverse(size(self%covariance_,1)),cutoff
    call symmetric_eigen(self%covariance_,values,vectors)
    cutoff=epsilon(1._dp)*size(values)*maxval(abs(values)); inverse=0
    where(values>cutoff) inverse=1/max(values,tiny(1._dp))
    self%precision_=matmul(vectors*spread(inverse,1,size(values)),transpose(vectors))
  end subroutine
  subroutine shrink(self,amount)
    class(empirical_covariance), intent(inout) :: self
    real(dp), intent(in) :: amount
    real(dp) :: mu
    integer :: j,p
    call require(amount>=0.and.amount<=1,'invalid covariance shrinkage')
    p=size(self%covariance_,1); mu=0
    do j=1,p
      mu=mu+self%covariance_(j,j)/p
    end do
    self%covariance_=(1-amount)*self%covariance_
    do j=1,p
      self%covariance_(j,j)=self%covariance_(j,j)+amount*mu
    end do
    call set_precision(self)
  end subroutine
  subroutine shrunk_fit(self,x)
    class(shrunk_covariance), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call empirical_fit(self,x); call shrink(self,self%shrinkage)
  end subroutine
  subroutine ledoit_fit(self,x)
    class(ledoit_wolf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp)::centered(size(x,1),size(x,2)),mu
    integer::j,p
    call require(self%block_size>=1,'Ledoit-Wolf block_size must be positive')
    call check_x(x);self%location_=sum(x,dim=1)/size(x,1)
    if(self%assume_centered)self%location_=0
    centered=x-spread(self%location_,1,size(x,1))
    self%covariance_=matmul(transpose(centered),centered)/size(x,1)
    self%shrinkage_=ledoit_wolf_shrinkage(x,self%assume_centered,self%block_size)
    call require(ieee_is_finite(self%shrinkage_),'Ledoit-Wolf shrinkage is nonfinite')
    ! _ledoit_wolf applies the computed coefficient without a range clamp.
    ! Roundoff can produce a tiny negative value for one-sample rank-one data.
    p=size(x,2);mu=0
    do j=1,p
      mu=mu+self%covariance_(j,j)
    end do
    mu=mu/p;self%covariance_=(1-self%shrinkage_)*self%covariance_
    do j=1,p
      self%covariance_(j,j)=self%covariance_(j,j)+self%shrinkage_*mu
    end do
    call set_precision(self)
  end subroutine
  subroutine oas_fit(self,x)
    class(oas), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: mu,alpha,den
    integer :: p,j
    call empirical_fit(self,x); p=size(x,2); self%shrinkage_=0
    if(p==1) return
    mu=0
    do j=1,p
      mu=mu+self%covariance_(j,j)/p
    end do
    alpha=sum(self%covariance_**2)/(p*p); den=(size(x,1)+1)*(alpha-mu**2/p)
    self%shrinkage_=1
    if(den>0) self%shrinkage_=min(1._dp,(alpha+mu**2)/den)
    call shrink(self,self%shrinkage_)
  end subroutine
  function cov_mahalanobis(self,x) result(distances)
    class(empirical_covariance), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: distances(size(x,1)),centered(size(x,1),size(x,2))
    call require(allocated(self%precision_),'covariance estimator is not fitted')
    call check_xy(transpose(x),size(self%location_)); centered=x-spread(self%location_,1,size(x,1))
    distances=max(0._dp,sum(matmul(centered,self%precision_)*centered,dim=2))
  end function
  real(dp) function cov_score(self,x) result(v)
    class(empirical_covariance), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: values(size(x,2)),vectors(size(x,2),size(x,2)),distances(size(x,1))
    distances=self%mahalanobis(x); call symmetric_eigen(self%precision_,values,vectors)
    call require(all(values>0),'log likelihood undefined for singular precision')
    v=0.5_dp*(sum(log(values))-size(x,2)*log(2*acos(-1._dp))-sum(distances)/size(x,1))
  end function
  real(dp) function cov_error(self,comp_cov,norm,scaling,squared) result(v)
    class(empirical_covariance), intent(in) :: self
    real(dp), intent(in) :: comp_cov(:,:)
    character(*), intent(in), optional :: norm
    logical, intent(in), optional :: scaling,squared
    real(dp), allocatable :: error(:,:),values(:),vectors(:,:)
    integer :: p
    logical :: scale,square
    character(16) :: kind
    call require(allocated(self%covariance_),'covariance estimator is not fitted')
    call require(all(shape(comp_cov)==shape(self%covariance_)),'covariance error shape mismatch')
    kind='frobenius'; if(present(norm)) kind=norm
    scale=.true.; square=.true.
    if(present(scaling)) scale=scaling
    if(present(squared)) square=squared
    error=comp_cov-self%covariance_; p=size(error,1)
    select case(trim(kind))
    case('frobenius'); v=sum(error**2)
    case('spectral')
      allocate(values(p),vectors(p,p)); call symmetric_eigen(matmul(transpose(error),error),values,vectors)
      v=maxval(values)
    case default; call require(.false.,'unknown covariance error norm')
    end select
    if(scale) v=v/p
    if(.not.square) v=sqrt(v)
  end function
end module
