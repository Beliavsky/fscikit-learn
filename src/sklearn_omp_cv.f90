! Adapted from sklearn 1.6.1 linear_model/_omp.py (BSD-3-Clause).
! Full upstream notice: LICENSES/scikit-learn-COPYING.
module sklearn_omp_cv
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,solve_lower,solve_upper
  use sklearn_metrics, only: r2_score
  implicit none
  private
  public :: orthogonal_matching_pursuit_cv
  type :: orthogonal_matching_pursuit_cv
    integer :: cv=5,max_iter=0,n_nonzero_coefs_=0,n_iter_=0,n_features_in_=0,n_splits_=0
    logical :: fit_intercept=.true.
    integer, allocatable :: test_fold(:),fold_n_iter_(:),fold_test_sizes_(:),fold_train_sizes_(:)
    logical, allocatable :: train_mask(:,:),test_mask(:,:)
    real(dp), allocatable :: coef_(:),mse_path_(:,:)
    real(dp) :: intercept_=0,best_mse_=0
  contains
    procedure :: fit=>omp_cv_fit
    procedure :: predict=>omp_cv_predict
    procedure :: score=>omp_cv_score
  end type
contains
  subroutine coefficient_path(x,y,limit,path,steps)
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in) :: limit
    real(dp), allocatable, intent(out) :: path(:,:)
    integer, intent(out) :: steps
    real(dp) :: a(size(x,1),size(x,2)),alpha(size(x,2)),residual(size(y))
    real(dp) :: l(limit,limit),row(limit),gamma(limit),column(size(x,1)),value,diagonal
    integer :: order(size(x,2)),k,j,swap_index,p
    p=size(x,2); a=x; alpha=matmul(transpose(a),y); residual=y
    order=[(j,j=1,p)]; allocate(path(p,limit)); path=0; l=0; steps=0
    do k=1,limit
      j=maxloc(abs(matmul(transpose(a),residual)),dim=1)
      if(j<k.or.alpha(j)**2<epsilon(1._dp)) exit
      if(k>1) then
        row(:k-1)=solve_lower(l(:k-1,:k-1),matmul(transpose(a(:,:k-1)),a(:,j)))
        diagonal=sqrt(sum(a(:,j)**2))**2-sqrt(sum(row(:k-1)**2))**2
        if(diagonal<=epsilon(1._dp)) exit
        l(k,:k-1)=row(:k-1); l(k,k)=sqrt(diagonal)
      else
        l(k,k)=sqrt(sum(a(:,j)**2))
      end if
      column=a(:,k); a(:,k)=a(:,j); a(:,j)=column
      value=alpha(k); alpha(k)=alpha(j); alpha(j)=value
      swap_index=order(k); order(k)=order(j); order(j)=swap_index
      gamma(:k)=solve_upper(transpose(l(:k,:k)),solve_lower(l(:k,:k),alpha(:k)))
      path(order(:k),k)=gamma(:k)
      residual=y-matmul(a(:,:k),gamma(:k)); steps=k
    end do
    path=path(:,:steps)
  end subroutine

  subroutine gram_path(x,y,limit,path,steps)
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in) :: limit
    real(dp), allocatable, intent(out) :: path(:,:)
    integer, intent(out) :: steps
    real(dp) :: gram(size(x,2),size(x,2)),xy(size(x,2)),alpha(size(x,2)),column(size(x,2))
    real(dp) :: l(limit,limit),row(limit),gamma(limit),diagonal,value
    integer :: p,k,j,swap_index,order(size(x,2))
    p=size(x,2); gram=matmul(transpose(x),x); xy=matmul(transpose(x),y); alpha=xy
    order=[(j,j=1,p)]; allocate(path(p,limit)); path=0; l=0; steps=0
    do k=1,limit
      j=maxloc(abs(alpha),dim=1)
      if(j<k.or.alpha(j)**2<epsilon(1._dp)) exit
      if(k>1) then
        row(:k-1)=solve_lower(l(:k-1,:k-1),gram(j,:k-1))
        diagonal=gram(j,j)-sqrt(sum(row(:k-1)**2))**2
        if(diagonal<=epsilon(1._dp)) exit
        l(k,:k-1)=row(:k-1); l(k,k)=sqrt(diagonal)
      else
        l(k,k)=sqrt(gram(j,j))
      end if
      column=gram(k,:); gram(k,:)=gram(j,:); gram(j,:)=column
      column=gram(:,k); gram(:,k)=gram(:,j); gram(:,j)=column
      value=xy(k); xy(k)=xy(j); xy(j)=value
      swap_index=order(k); order(k)=order(j); order(j)=swap_index
      gamma(:k)=solve_upper(transpose(l(:k,:k)),solve_lower(l(:k,:k),xy(:k)))
      path(order(:k),k)=gamma(:k); steps=k
      alpha=xy-matmul(gram(:,:k),gamma(:k))
    end do
    path=path(:,:steps)
  end subroutine

  subroutine make_partitions(self,n,train,test)
    class(orthogonal_matching_pursuit_cv), intent(inout) :: self
    integer, intent(in) :: n
    logical, allocatable, intent(out) :: train(:,:),test(:,:)
    integer, allocatable :: labels(:),ids(:)
    integer :: f,first,last,width
    call require(allocated(self%train_mask).eqv.allocated(self%test_mask),'OMP CV requires both fold masks')
    if(allocated(self%train_mask)) then
      call require(.not.allocated(self%test_fold),'OMP CV fold masks and test_fold are mutually exclusive')
      call require(size(self%train_mask,1)==n.and.size(self%test_mask,1)==n,'OMP CV mask row count mismatch')
      call require(size(self%train_mask,2)>0.and.size(self%train_mask,2)==size(self%test_mask,2), &
        'OMP CV mask fold count mismatch')
      train=self%train_mask; test=self%test_mask
    else if(allocated(self%test_fold)) then
      call require(size(self%test_fold)==n,'OMP CV fold map length mismatch')
      ids=pack(self%test_fold,self%test_fold/=-1)
      call require(size(ids)>0,'OMP CV fold map has no test samples')
      labels=unique_labels(ids); allocate(train(n,size(labels)),test(n,size(labels)))
      do f=1,size(labels)
        test(:,f)=self%test_fold==labels(f); train(:,f)=.not.test(:,f)
      end do
    else
      call require(self%cv>=2.and.self%cv<=n,'OMP CV needs between two and N folds')
      allocate(train(n,self%cv),test(n,self%cv)); test=.false.; first=1
      do f=1,self%cv
        width=n/self%cv
        if(f<=mod(n,self%cv)) width=width+1
        last=first+width-1; test(first:last,f)=.true.; first=last+1
      end do
      train=.not.test
    end if
    self%n_splits_=size(train,2)
    self%fold_train_sizes_=count(train,dim=1); self%fold_test_sizes_=count(test,dim=1)
    call require(all(self%fold_train_sizes_>0).and.all(self%fold_test_sizes_>0), &
      'OMP CV folds require nonempty train and test sets')
  end subroutine

  subroutine omp_cv_fit(self,x,y)
    class(orthogonal_matching_pursuit_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    logical, allocatable :: train(:,:),test(:,:)
    integer, allocatable :: ids(:),ti(:),vi(:)
    real(dp), allocatable :: a(:,:),b(:),xm(:),path(:,:),errors(:,:),prediction(:,:),mean_mse(:)
    real(dp) :: ym
    integer :: n,p,limit,f,j,steps,common_steps
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    n=size(x,1); p=size(x,2)
    call require(p>=2,'OMP CV requires at least two features')
    call require(self%max_iter>=0.and.self%max_iter<=p,'OMP CV max_iter must be between zero and P')
    limit=self%max_iter
    if(limit==0) limit=min(max(p/10,5),p)
    call make_partitions(self,n,train,test); ids=[(j,j=1,n)]
    allocate(errors(limit,self%n_splits_)); errors=0
    if(allocated(self%fold_n_iter_)) deallocate(self%fold_n_iter_)
    allocate(self%fold_n_iter_(self%n_splits_)); common_steps=limit
    do f=1,self%n_splits_
      ti=pack(ids,train(:,f)); vi=pack(ids,test(:,f))
      xm=sum(x(ti,:),dim=1)/real(size(ti),dp); ym=sum(y(ti))/real(size(ti),dp)
      if(.not.self%fit_intercept) then
        xm=0; ym=0
      end if
      a=x(ti,:)-spread(xm,1,size(ti)); b=y(ti)-ym
      call coefficient_path(a,b,limit,path,steps)
      self%fold_n_iter_(f)=steps; common_steps=min(common_steps,steps)
      prediction=matmul(x(vi,:)-spread(xm,1,size(vi)),path)
      do j=1,steps
        errors(j,f)=sum((prediction(:,j)-(y(vi)-ym))**2)/real(size(vi),dp)
      end do
    end do
    call require(common_steps>0,'OMP CV folds produced no active steps')
    self%mse_path_=errors(:common_steps,:)
    mean_mse=sum(self%mse_path_,dim=2)/real(self%n_splits_,dp)
    self%n_nonzero_coefs_=minloc(mean_mse,dim=1); self%best_mse_=mean_mse(self%n_nonzero_coefs_)
    xm=sum(x,dim=1)/real(n,dp); ym=sum(y)/real(n,dp)
    if(.not.self%fit_intercept) then
      xm=0; ym=0
    end if
    a=x-spread(xm,1,n); b=y-ym
    if(n>p) then
      call gram_path(a,b,self%n_nonzero_coefs_,path,self%n_iter_)
    else
      call coefficient_path(a,b,self%n_nonzero_coefs_,path,self%n_iter_)
    end if
    self%coef_=spread(0._dp,1,p)
    if(self%n_iter_>0) self%coef_=path(:,self%n_iter_)
    self%intercept_=ym-dot_product(xm,self%coef_); self%n_features_in_=p
  end subroutine

  function omp_cv_predict(self,x) result(y)
    class(orthogonal_matching_pursuit_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%coef_),'OMP CV model is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'OMP CV feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function omp_cv_score(self,x,y) result(score)
    class(orthogonal_matching_pursuit_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: prediction(:)
    prediction=self%predict(x); call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    score=r2_score(y,prediction)
  end function
end module
