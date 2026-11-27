module sklearn_quantile
  use sklearn_common, only: dp,require,check_x,check_xy
  use sklearn_metrics, only: r2_score
  use sklearn_sparse,only:csr_matrix
  use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_total
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: quantile_regressor
  ! sklearn linear_model/_quantile.py objective; independent dense primal simplex.
  type :: quantile_regressor
    real(dp) :: quantile=.5_dp,alpha=1,tol=1.e-10_dp
    logical :: fit_intercept=.true.,converged_=.false.
    integer :: max_iter=10000,n_iter_=0,n_features_in_=0
    real(dp) :: intercept_=0,objective_=0,dual_gap_=0,optimality_=0
    real(dp), allocatable :: coef_(:),dual_coef_(:)
  contains
    procedure :: fit_dense=>quantile_fit,fit_csr,fit_scalar_dense,fit_scalar_csr
    procedure :: fit_column_dense,fit_column_csr,fit_column_scalar_dense,fit_column_scalar_csr
    generic :: fit=>fit_dense,fit_csr,fit_scalar_dense,fit_scalar_csr, &
      fit_column_dense,fit_column_csr,fit_column_scalar_dense,fit_column_scalar_csr
    procedure :: predict_dense=>quantile_predict,predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure :: score_dense=>quantile_score,score_csr,score_column_dense,score_column_csr
    generic :: score=>score_dense,score_csr,score_column_dense,score_column_csr
  end type
contains
  subroutine check_sparse(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'quantile sparse design must be nonempty')
    call require(all(ieee_is_finite(x%data)),'quantile sparse design must be finite')
  end subroutine
  subroutine fit_csr(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call fit_common(self,y,sample_weight,x_sparse=x)
  end subroutine
  subroutine fit_scalar_dense(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight
    call fit_common(self,y,spread(sample_weight,1,size(y)),x=x)
  end subroutine
  subroutine fit_scalar_csr(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    call fit_common(self,y,spread(sample_weight,1,size(y)),x_sparse=x)
  end subroutine
  subroutine fit_column_dense(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'QuantileRegressor requires one target column')
    call fit_common(self,y(:,1),sample_weight,x=x)
  end subroutine
  subroutine fit_column_csr(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'QuantileRegressor requires one target column')
    call fit_common(self,y(:,1),sample_weight,x_sparse=x)
  end subroutine
  subroutine fit_column_scalar_dense(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:),sample_weight
    call require(size(y,2)==1,'QuantileRegressor requires one target column')
    call fit_common(self,y(:,1),spread(sample_weight,1,size(y,1)),x=x)
  end subroutine
  subroutine fit_column_scalar_csr(self,x,y,sample_weight)
    class(quantile_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    call require(size(y,2)==1,'QuantileRegressor requires one target column')
    call fit_common(self,y(:,1),spread(sample_weight,1,size(y,1)),x_sparse=x)
  end subroutine
  subroutine quantile_fit(self,x,y,sample_weight)
    class(quantile_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call fit_common(self,y,sample_weight,x=x)
  end subroutine
  subroutine fit_common(self,y,sample_weight,x,x_sparse)
    class(quantile_regressor),intent(inout)::self
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),x(:,:)
    type(csr_matrix),optional,intent(in)::x_sparse
    real(dp), allocatable :: weights(:),a(:,:),cost(:),reduced(:),solution(:),scale(:),target(:),dual(:),residual(:)
    real(dp) :: total,yscale,signrow,pivot,factor,ratio,best,pivot_tol,feas_tol
    integer, allocatable :: indices(:),basis(:)
    integer :: n,p,q,m,i,j,k,enter,leave,offset
    if(present(x))then
      call check_xy(x,size(y));p=size(x,2)
    else
      call require(present(x_sparse),'quantile design is absent')
      call check_sparse(x_sparse)
      call require(x_sparse%n_rows==size(y),'quantile target length mismatch');p=x_sparse%n_cols
    end if
    call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%quantile),'quantile must be finite and strictly between zero and one')
    call require(self%quantile>0.and.self%quantile<1,'quantile must be finite and strictly between zero and one')
    call require(ieee_is_finite(self%alpha),'quantile alpha must be finite and nonnegative')
    call require(self%alpha>=0,'quantile alpha must be finite and nonnegative')
    call require(ieee_is_finite(self%tol),'quantile tolerance must be finite and positive')
    call require(self%tol>0,'quantile tolerance must be finite and positive')
    call require(self%max_iter>=1,'quantile max_iter must be positive')
    weights=spread(1._dp,1,size(y))
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'quantile sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'quantile weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'quantile weights must be finite and nonnegative')
      weights=sample_weight
    end if
    total=ieee_total(weights)
    call require(ieee_is_finite(total),'quantile weights must have finite sum')
    if(total==0)then
      self%coef_=spread(0._dp,1,p);self%intercept_=0;self%dual_coef_=spread(0._dp,1,size(y))
      self%n_features_in_=p;self%n_iter_=0;self%converged_=.true.
      self%objective_=0;self%dual_gap_=0;self%optimality_=0;return
    end if
    weights=weights/total
    indices=pack([(i,i=1,size(y))],weights>0)
    weights=weights(indices); n=size(indices); offset=merge(1,0,self%fit_intercept); q=p+offset
    m=2*q+2*n; allocate(a(n,m+1),cost(m),reduced(m),solution(m),basis(n),scale(q),target(n),dual(n))
    a=0; cost=0; scale=1; target=y(indices); yscale=max(1._dp,maxval(abs(target))); target=target/yscale
    if(self%fit_intercept) a(:,1)=1
    if(present(x))then
      a(:,offset+1:q)=x(indices,:)
    else
      do i=1,n
        do k=x_sparse%indptr(indices(i)),x_sparse%indptr(indices(i)+1)-1
          j=x_sparse%indices(k)+offset
          a(i,j)=ieee_difference(a(i,j),-x_sparse%data(k))
        end do
      end do
    end if
    call require(all(ieee_is_finite(a(:,offset+1:q))),'quantile design accumulation must be finite')
    do j=1,p
      scale(j+offset)=max(1._dp,maxval(abs(a(:,j+offset))))
      a(:,j+offset)=a(:,j+offset)/scale(j+offset)
      cost(j+offset)=self%alpha/scale(j+offset)
    end do
    a(:,q+1:2*q)=-a(:,:q); cost(q+1:2*q)=cost(:q)
    cost(2*q+1:2*q+n)=weights*self%quantile
    cost(2*q+n+1:m)=weights*(1-self%quantile)
    do i=1,n
      a(i,2*q+i)=1; a(i,2*q+n+i)=-1
      signrow=1
      basis(i)=2*q+i
      if(target(i)<0) then
        signrow=-1; basis(i)=2*q+n+i
      end if
      a(i,:m)=signrow*a(i,:m); a(i,m+1)=abs(target(i))
    end do
    self%n_iter_=0; self%converged_=.false.
    ! Residual columns give a feasible initial basis without artificial variables.
    ! Bland's entering/leaving index rules avoid cycling on degenerate LPs.
    do
      reduced=cost-matmul(cost(basis),a(:,:m))
      reduced(basis)=0
      enter=0
      do j=1,m
        if(reduced(j)<-self%tol) then
          enter=j
          exit
        end if
      end do
      if(enter==0) then
        self%converged_=.true.
        exit
      end if
      if(self%n_iter_>=self%max_iter) exit
      pivot_tol=32*epsilon(1._dp)*max(1._dp,maxval(abs(a(:,enter))))
      leave=0; best=huge(1._dp)
      do i=1,n
        if(a(i,enter)<=pivot_tol) cycle
        ratio=max(0._dp,a(i,m+1))/a(i,enter)
        if(ratio<best) then
          best=ratio; leave=i
        else if(ratio==best.and.leave>0) then
          if(basis(i)<basis(leave)) leave=i
        end if
      end do
      ! A bounded quantile LP always has a leaving row. Report nonconvergence
      ! if numerical scaling prevents a trustworthy pivot.
      if(leave==0) exit
      pivot=a(leave,enter); a(leave,:)=a(leave,:)/pivot; a(leave,enter)=1
      do i=1,n
        if(i==leave) cycle
        factor=a(i,enter); a(i,:)=a(i,:)-factor*a(leave,:); a(i,enter)=0
      end do
      basis(leave)=enter; self%n_iter_=self%n_iter_+1
      feas_tol=256*epsilon(1._dp)*max(1._dp,maxval(abs(a(:,m+1))))
      if(minval(a(:,m+1)) < -feas_tol) exit
      a(:,m+1)=max(0._dp,a(:,m+1))
    end do
    solution=0; solution(basis)=a(:,m+1)
    self%coef_=yscale*(solution(offset+1:q)-solution(q+offset+1:2*q))/scale(offset+1:q)
    self%intercept_=0
    if(self%fit_intercept) self%intercept_=yscale*(solution(1)-solution(q+1))
    self%n_features_in_=p
    reduced=cost-matmul(cost(basis),a(:,:m)); reduced(basis)=0
    self%optimality_=max(0._dp,-minval(reduced))
    dual=matmul(cost(basis),a(:,2*q+1:2*q+n))
    self%dual_coef_=spread(0._dp,1,size(y)); self%dual_coef_(indices)=dual
    residual=y(indices)-self%intercept_
    if(present(x))then
      residual=residual-matmul(x(indices,:),self%coef_)
    else
      do i=1,n
        do k=x_sparse%indptr(indices(i)),x_sparse%indptr(indices(i)+1)-1
          residual(i)=ieee_difference(residual(i),ieee_product(x_sparse%data(k),self%coef_(x_sparse%indices(k))))
        end do
      end do
    end if
    self%objective_=sum(weights*max(self%quantile*residual,(self%quantile-1)*residual))+ &
      self%alpha*sum(abs(self%coef_))
    self%dual_gap_=max(0._dp,self%objective_-dot_product(y(indices),dual))
  end subroutine
  function quantile_predict(self,x) result(y)
    class(quantile_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%coef_),'QuantileRegressor is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'quantile feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function quantile_score(self,x,y,sample_weight) result(v)
    class(quantile_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    v=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  function predict_csr(self,x)result(y)
    class(quantile_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::y(x%n_rows)
    integer::i,k
    call require(allocated(self%coef_),'QuantileRegressor is not fitted');call check_sparse(x)
    call require(x%n_cols==self%n_features_in_,'quantile feature count mismatch')
    y=self%intercept_
    do i=1,x%n_rows
      do k=x%indptr(i),x%indptr(i+1)-1
        y(i)=ieee_difference(y(i),-ieee_product(x%data(k),self%coef_(x%indices(k))))
      end do
    end do
  end function
  real(dp)function score_csr(self,x,y,sample_weight)result(value)
    class(quantile_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function score_column_dense(self,x,y,sample_weight)result(value)
    class(quantile_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'QuantileRegressor score requires one target column')
    value=r2_score(y(:,1),self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function score_column_csr(self,x,y,sample_weight)result(value)
    class(quantile_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'QuantileRegressor score requires one target column')
    value=r2_score(y(:,1),self%predict(x),sample_weight=sample_weight)
  end function
end module
