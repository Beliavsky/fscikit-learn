! Multi-task coordinate descent contracts: sklearn 1.6.1 _coordinate_descent.py/_cd_fast.pyx.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_multitask
  use sklearn_common, only: dp,require,check_x,check_xy,argsort,unique_labels
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: multi_task_elastic_net,multi_task_lasso
  public :: multi_task_elastic_net_cv,multi_task_lasso_cv
  public :: multitask_path_step
  type :: multi_task_elastic_net
    real(dp) :: alpha=1,l1_ratio=.5_dp,tol=1.e-4_dp
    integer :: max_iter=1000,n_iter_=0,n_features_in_=0,random_state=0
    character(6) :: selection='cyclic'
    logical :: fit_intercept=.true.,warm_start=.false.,converged_=.false.
    real(dp) :: dual_gap_=0,eps_=0
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>multitask_fit
    procedure :: predict=>multitask_predict
    procedure :: score=>multitask_score
  end type
  type, extends(multi_task_elastic_net) :: multi_task_lasso
  contains
    procedure :: fit=>multitask_lasso_fit
  end type
  type, extends(multi_task_elastic_net) :: multi_task_elastic_net_cv
    integer :: cv=5,n_alphas=100,n_splits_=0
    real(dp) :: eps=1.e-3_dp,alpha_=0,l1_ratio_=0,best_mse_=0
    real(dp), allocatable :: alphas(:),l1_ratios(:),alphas_(:,:),mse_path_(:,:,:),fold_dual_gaps_(:,:,:)
    integer, allocatable :: test_fold(:),fold_labels_(:),fold_test_sizes_(:),fold_n_iter_(:,:,:)
    logical :: all_cv_converged_=.false.
  contains
    procedure :: fit=>multitask_cv_fit
  end type
  type, extends(multi_task_elastic_net_cv) :: multi_task_lasso_cv
  contains
    procedure :: fit=>multitask_lasso_cv_fit
  end type
contains
  ! Shared by the public coordinate-descent path without resetting its RNG.
  subroutine multitask_path_step(model,x,y,rng)
    type(multi_task_elastic_net),intent(inout)::model
    real(dp),intent(in)::x(:,:),y(:,:)
    type(legacy_random_state),intent(inout)::rng
    call multitask_solve(model,x,y,model%l1_ratio,rng)
  end subroutine
  integer function random_coordinate(state,p) result(coordinate)
    integer(int64), intent(inout) :: state
    integer, intent(in) :: p
    integer(int64), parameter :: mask32=int(z'FFFFFFFF',int64),mask31=int(z'7FFFFFFF',int64)
    if(state==0) state=1
    state=iand(ieor(state,ishft(state,13)),mask32)
    state=ieor(state,ishft(state,-17))
    state=iand(ieor(state,ishft(state,5)),mask32)
    coordinate=int(modulo(iand(state,mask31),int(p,int64)))+1
  end function
  subroutine multitask_fit(self,x,y)
    class(multi_task_elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call require(ieee_is_finite(self%l1_ratio),'nonfinite MultiTask l1_ratio')
    call require(self%l1_ratio>=0.and.self%l1_ratio<=1,'MultiTask l1_ratio must be in [0,1]')
    call multitask_solve(self,x,y,self%l1_ratio)
  end subroutine

  subroutine multitask_lasso_fit(self,x,y)
    class(multi_task_lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call multitask_solve(self,x,y,1._dp)
  end subroutine

  subroutine multitask_solve(self,x,y,mix,path_rng)
    class(multi_task_elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:),mix
    type(legacy_random_state), optional, intent(inout) :: path_rng
    type(legacy_random_state) :: rng
    integer(int64) :: random_state
    real(dp), allocatable :: a(:,:),b(:,:),xm(:),ym(:),residual(:,:),norms(:),old(:),z(:),gradient(:,:)
    real(dp) :: l1,l2,znorm,change,largest,dualnorm,scale,gap,rnorm2,l21
    integer :: n,p,t,j,iteration,coordinate
    call check_xy(x,size(y,1)); call check_x(y)
    call require(ieee_is_finite(self%alpha),'nonfinite MultiTask alpha')
    call require(self%alpha>=0,'MultiTask alpha must be nonnegative')
    call require(self%selection=='cyclic'.or.self%selection=='random','unknown MultiTask coordinate selection')
    call require(self%random_state>=0,'MultiTask seed must be nonnegative')
    if(present(path_rng)) then
      random_state=int(path_rng%integer_index(2147483647)-1,int64)
    else
      call rng%seed(self%random_state); random_state=int(rng%integer_index(2147483647)-1,int64)
    end if
    call require(ieee_is_finite(self%tol),'nonfinite MultiTask tolerance')
    call require(self%tol>=0,'MultiTask tolerance must be nonnegative')
    call require(self%max_iter>0,'MultiTask max_iter must be positive')
    n=size(x,1); p=size(x,2); t=size(y,2)
    xm=sum(x,dim=1)/real(n,dp); ym=sum(y,dim=1)/real(n,dp)
    if(.not.self%fit_intercept) then
      xm=0; ym=0
    end if
    a=x-spread(xm,1,n); b=y-spread(ym,1,n)
    if(self%warm_start.and.allocated(self%coef_)) then
      call require(size(self%coef_,1)==t.and.size(self%coef_,2)==p,'MultiTask warm-start dimensions changed')
    else
      self%coef_=spread(spread(0._dp,1,t),2,p)
    end if
    residual=b-matmul(a,transpose(self%coef_)); norms=sum(a*a,dim=1)
    allocate(old(t),z(t)); l1=self%alpha*mix*n; l2=self%alpha*(1-mix)*n
    self%eps_=self%tol*sum(b*b); self%converged_=.false.
    do iteration=1,self%max_iter
      change=0; largest=0
      do coordinate=1,p
        j=coordinate
        if(self%selection=='random') j=random_coordinate(random_state,p)
        if(norms(j)==0) cycle
        old=self%coef_(:,j)
        residual=residual+spread(a(:,j),2,t)*spread(old,1,n)
        z=matmul(a(:,j),residual); znorm=sqrt(sum(z*z))
        self%coef_(:,j)=0
        if(znorm>l1) self%coef_(:,j)=z*((1-l1/znorm)/(norms(j)+l2))
        residual=residual-spread(a(:,j),2,t)*spread(self%coef_(:,j),1,n)
        change=max(change,maxval(abs(self%coef_(:,j)-old)))
        largest=max(largest,maxval(abs(self%coef_(:,j))))
      end do
      if(largest==0.or.change<self%tol*largest.or.iteration==self%max_iter) then
        gradient=matmul(transpose(a),residual)-l2*transpose(self%coef_)
        dualnorm=maxval(sqrt(sum(gradient*gradient,dim=2)))
        scale=1
        if(dualnorm>l1) scale=l1/dualnorm
        rnorm2=sum(residual*residual); l21=sum(sqrt(sum(self%coef_**2,dim=1)))
        gap=.5_dp*rnorm2*(1+scale**2)+l1*l21-scale*sum(residual*b) &
          +.5_dp*l2*(1+scale**2)*sum(self%coef_**2)
        self%dual_gap_=gap/real(n,dp)
        if(gap<=self%eps_) then
          self%converged_=.true.; exit
        end if
      end if
    end do
    self%n_iter_=min(iteration,self%max_iter); self%n_features_in_=p
    self%intercept_=ym-matmul(self%coef_,xm)
  end subroutine

  function multitask_predict(self,x) result(y)
    class(multi_task_elastic_net), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call require(allocated(self%coef_),'MultiTask model is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'MultiTask feature count mismatch')
    y=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  real(dp) function multitask_score(self,x,y) result(score)
    class(multi_task_elastic_net), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: prediction(:,:)
    integer :: j
    prediction=self%predict(x); call check_xy(y,size(x,1))
    call require(size(y,2)==size(prediction,2),'MultiTask target count mismatch')
    score=0
    do j=1,size(y,2)
      score=score+r2_score(y(:,j),prediction(:,j))
    end do
    score=score/real(size(y,2),dp)
  end function
  subroutine multitask_cv_fit(self,x,y)
    class(multi_task_elastic_net_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call fit_cv(self,x,y,.false.)
  end subroutine
  subroutine multitask_lasso_cv_fit(self,x,y)
    class(multi_task_lasso_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call fit_cv(self,x,y,.true.)
  end subroutine
  subroutine fold_assignment(self,n,assignment)
    class(multi_task_elastic_net_cv), intent(inout) :: self
    integer, intent(in) :: n
    integer, allocatable, intent(out) :: assignment(:)
    integer, allocatable :: candidates(:),train(:),test(:)
    integer :: f,first,last,width
    if(allocated(self%test_fold)) then
      call require(size(self%test_fold)==n,'MultiTask CV fold map length mismatch')
      assignment=self%test_fold; candidates=pack(assignment,assignment/=-1)
      call require(size(candidates)>0,'MultiTask CV fold map has no test samples')
      self%fold_labels_=unique_labels(candidates)
    else
      call require(self%cv>=2.and.self%cv<=n,'MultiTask CV needs between two and N folds')
      self%fold_labels_=[(f-1,f=1,self%cv)]; allocate(assignment(n)); first=1
      do f=1,self%cv
        width=n/self%cv
        if(f<=mod(n,self%cv)) width=width+1
        last=first+width-1; assignment(first:last)=f-1; first=last+1
      end do
    end if
    self%n_splits_=size(self%fold_labels_)
    if(allocated(self%fold_test_sizes_)) deallocate(self%fold_test_sizes_)
    allocate(self%fold_test_sizes_(self%n_splits_))
    do f=1,self%n_splits_
      self%fold_test_sizes_(f)=count(assignment==self%fold_labels_(f))
      call require(self%fold_test_sizes_(f)>0.and.self%fold_test_sizes_(f)<n, &
        'MultiTask CV folds require nonempty train and test sets')
    end do
  end subroutine
  subroutine fit_cv(self,x,y,lasso)
    class(multi_task_elastic_net_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    logical, intent(in) :: lasso
    type(multi_task_elastic_net) :: model
    type(legacy_random_state) :: rng
    real(dp), allocatable :: ratios(:),grid(:),a(:,:),b(:,:),xm(:),ym(:),correlation(:,:),prediction(:,:)
    integer, allocatable :: assignment(:),train(:),test(:),ids(:),order(:)
    real(dp) :: alpha_max,mse
    integer :: n,p,t,nr,na,r,f,j,best_r,best_a
    call check_xy(x,size(y,1)); call check_x(y)
    call require(self%max_iter>0,'MultiTask max_iter must be positive')
    call require(ieee_is_finite(self%tol),'nonfinite MultiTask tolerance')
    call require(self%tol>=0,'MultiTask tolerance must be nonnegative')
    call require(ieee_is_finite(self%eps),'MultiTask CV eps must be finite')
    call require(self%eps>0,'MultiTask CV eps must be positive')
    call require(self%n_alphas>=1,'MultiTask CV n_alphas must be positive')
    call require(self%selection=='cyclic'.or.self%selection=='random','unknown MultiTask coordinate selection')
    call require(self%random_state>=0,'MultiTask seed must be nonnegative')
    if(lasso) then
      ratios=[1._dp]
    else if(allocated(self%l1_ratios)) then
      ratios=self%l1_ratios
    else
      ratios=[self%l1_ratio]
    end if
    call require(size(ratios)>0,'MultiTask CV needs at least one l1_ratio')
    call require(all(ieee_is_finite(ratios)),'MultiTask CV ratios must be finite')
    call require(all(ratios>=0).and.all(ratios<=1),'MultiTask CV ratios must be in [0,1]')
    n=size(x,1); p=size(x,2); t=size(y,2); nr=size(ratios)
    call fold_assignment(self,n,assignment); ids=[(j,j=1,n)]
    if(allocated(self%alphas)) then
      call require(size(self%alphas)>0,'MultiTask CV needs candidate alphas')
      call require(all(ieee_is_finite(self%alphas)),'MultiTask CV alphas must be finite')
      call require(all(self%alphas>=0),'MultiTask CV alphas must be nonnegative')
      na=size(self%alphas); order=argsort(-self%alphas); grid=self%alphas(order)
    else
      call require(all(ratios>0),'MultiTask CV zero l1_ratio requires explicit alphas')
      na=self%n_alphas
    end if
    if(allocated(self%alphas_)) deallocate(self%alphas_,self%mse_path_,self%fold_n_iter_,self%fold_dual_gaps_)
    allocate(self%alphas_(na,nr),self%mse_path_(na,self%n_splits_,nr), &
      self%fold_n_iter_(na,self%n_splits_,nr),self%fold_dual_gaps_(na,self%n_splits_,nr))
    xm=sum(x,dim=1)/real(n,dp); ym=sum(y,dim=1)/real(n,dp)
    if(.not.self%fit_intercept) then
      xm=0; ym=0
    end if
    a=x-spread(xm,1,n); b=y-spread(ym,1,n); correlation=matmul(transpose(a),b)
    do r=1,nr
      if(allocated(self%alphas)) then
        self%alphas_(:,r)=grid
      else
        alpha_max=maxval(sqrt(sum(correlation**2,dim=2)))/(real(n,dp)*ratios(r))
        if(alpha_max<=1.e-15_dp) then
          self%alphas_(:,r)=1.e-15_dp
        else
          self%alphas_(:,r)=[(alpha_max*exp(log(self%eps)*real(j-1,dp)/max(1,na-1)),j=1,na)]
        end if
      end if
    end do
    self%all_cv_converged_=.true.; best_r=1; best_a=1; self%best_mse_=huge(1._dp)
    do r=1,nr
      order=argsort(-self%alphas_(:,r)); grid=self%alphas_(order,r)
      do f=1,self%n_splits_
        train=pack(ids,assignment/=self%fold_labels_(f)); test=pack(ids,assignment==self%fold_labels_(f))
        xm=sum(x(train,:),dim=1)/real(size(train),dp); ym=sum(y(train,:),dim=1)/real(size(train),dp)
        if(.not.self%fit_intercept) then
          xm=0; ym=0
        end if
        a=x(train,:)-spread(xm,1,size(train)); b=y(train,:)-spread(ym,1,size(train))
        model=multi_task_elastic_net(); model%max_iter=self%max_iter; model%tol=self%tol
        model%fit_intercept=.false.; model%warm_start=.true.; model%selection=self%selection
        model%random_state=self%random_state; call rng%seed(self%random_state)
        do j=1,na
          model%alpha=grid(j); call multitask_solve(model,a,b,ratios(r),rng)
          prediction=matmul(x(test,:),transpose(model%coef_))+ &
            spread(ym-matmul(model%coef_,xm),1,size(test))
          self%mse_path_(j,f,r)=sum((prediction-y(test,:))**2)/(real(size(test),dp)*t)
          self%fold_n_iter_(j,f,r)=model%n_iter_; self%fold_dual_gaps_(j,f,r)=model%dual_gap_
          self%all_cv_converged_=self%all_cv_converged_.and.model%converged_
        end do
      end do
      do j=1,na
        mse=sum(self%mse_path_(j,:,r))/self%n_splits_
        if(mse<self%best_mse_) then
          self%best_mse_=mse; best_r=r; best_a=j
        end if
      end do
    end do
    self%alpha_=self%alphas_(best_a,best_r); self%l1_ratio_=ratios(best_r)
    self%alpha=self%alpha_; self%warm_start=.false.
    call multitask_solve(self,x,y,self%l1_ratio_)
  end subroutine
end module
