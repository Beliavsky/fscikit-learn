! Ridge classification contracts: sklearn 1.6.1 linear_model/_ridge.py.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_ridge_classification
  use sklearn_common, only: dp,require,check_xy,unique_labels,label_index,dense_svd
  use sklearn_metrics, only: accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: ridge_classifier,ridge_classifier_cv
  type :: ridge_classifier
    real(dp) :: alpha=1
    logical :: fit_intercept=.true.,class_weight_balanced=.false.
    integer :: n_features_in_=0,rank_=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>ridge_class_fit
    procedure :: decision_function=>ridge_class_decision
    procedure :: predict=>ridge_class_predict
    procedure :: score=>ridge_class_score
  end type
  type, extends(ridge_classifier) :: ridge_classifier_cv
    real(dp), allocatable :: alphas(:),cv_scores_(:),cv_results_(:,:,:)
    real(dp) :: alpha_=0,best_score_=0
    logical :: store_cv_results=.false.
  contains
    procedure :: fit=>ridge_class_cv_fit
  end type
contains
  subroutine ridge_class_setup(self,x,y,w,xmean,ymean,b,s,u,v,sample_weight,class_weight)
    class(ridge_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable, intent(out) :: w(:),xmean(:),ymean(:),b(:,:),s(:),u(:,:),v(:,:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    real(dp), allocatable :: targets(:,:),a(:,:),cw(:)
    real(dp) :: total,cutoff
    integer, allocatable :: encoded(:)
    integer :: n,p,c,k,i
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    self%classes_=unique_labels(y); c=size(self%classes_)
    call require(c>=2,'RidgeClassifier requires at least two classes')
    encoded=[(label_index(self%classes_,y(i)),i=1,n)]
    w=spread(1._dp,1,n); cw=spread(1._dp,1,c)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'RidgeClassifier sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite RidgeClassifier sample weights')
      call require(all(sample_weight>=0),'negative RidgeClassifier sample weights')
      w=sample_weight
    end if
    if(present(class_weight)) then
      call require(.not.self%class_weight_balanced,'choose balanced or explicit RidgeClassifier class weights')
      call require(size(class_weight)==c,'RidgeClassifier class weight count mismatch')
      call require(all(ieee_is_finite(class_weight)),'nonfinite RidgeClassifier class weights')
      call require(all(class_weight>=0),'negative RidgeClassifier class weights')
      cw=class_weight
    else if(self%class_weight_balanced) then
      do k=1,c
        cw(k)=real(n,dp)/(real(c,dp)*count(encoded==k))
      end do
    end if
    w=w*cw(encoded); total=sum(w)
    call require(ieee_is_finite(total),'nonfinite RidgeClassifier effective weights')
    call require(total>0,'RidgeClassifier needs positive total weight')
    k=c
    if(c==2) k=1
    allocate(targets(n,k)); targets=-1
    do i=1,n
      if(c==2) then
        if(encoded(i)==2) targets(i,1)=1
      else
        targets(i,encoded(i))=1
      end if
    end do
    xmean=spread(0._dp,1,p); ymean=spread(0._dp,1,k)
    if(self%fit_intercept) then
      xmean=matmul(w,x)/total; ymean=matmul(w,targets)/total
    end if
    a=(x-spread(xmean,1,n))*spread(sqrt(w),2,p)
    b=(targets-spread(ymean,1,n))*spread(sqrt(w),2,k)
    allocate(s(p),u(n,p),v(p,p)); call dense_svd(a,s,u,v)
    cutoff=epsilon(1._dp)*max(n,p)*maxval(s)
    self%rank_=count(s>cutoff); self%n_features_in_=p
    ! Match dense_svd's rank truncation in both coefficient and LOO calculations.
    do i=1,p
      if(s(i)<=cutoff) then
        s(i)=0; u(:,i)=0
      end if
    end do
  end subroutine

  subroutine ridge_class_coefficients(self,alpha,s,u,v,b,xmean,ymean)
    class(ridge_classifier), intent(inout) :: self
    real(dp), intent(in) :: alpha,s(:),u(:,:),v(:,:),b(:,:),xmean(:),ymean(:)
    real(dp), allocatable :: projection(:,:)
    integer :: j
    projection=matmul(transpose(u),b)
    do j=1,size(s)
      if(s(j)>0) then
        projection(j,:)=projection(j,:)*(s(j)/(s(j)**2+alpha))
      else
        projection(j,:)=0
      end if
    end do
    self%coef_=transpose(matmul(v,projection))
    self%intercept_=ymean-matmul(self%coef_,xmean)
  end subroutine

  subroutine ridge_class_fit(self,x,y,sample_weight,class_weight)
    class(ridge_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    real(dp), allocatable :: w(:),xm(:),ym(:),b(:,:),s(:),u(:,:),v(:,:)
    call require(ieee_is_finite(self%alpha),'nonfinite RidgeClassifier alpha')
    call require(self%alpha>=0,'RidgeClassifier alpha must be nonnegative')
    call ridge_class_setup(self,x,y,w,xm,ym,b,s,u,v,sample_weight,class_weight)
    call ridge_class_coefficients(self,self%alpha,s,u,v,b,xm,ym)
  end subroutine

  subroutine ridge_class_cv_fit(self,x,y,sample_weight,class_weight)
    class(ridge_classifier_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    real(dp), allocatable :: w(:),xm(:),ym(:),b(:,:),s(:),u(:,:),v(:,:),projection(:,:),outside(:,:)
    real(dp), allocatable :: remainder(:),inverse(:),dual(:,:),diagonal(:),errors(:,:)
    real(dp) :: alpha,score,total
    integer :: n,p,k,j,i,best,positive
    logical :: full_span
    if(.not.allocated(self%alphas)) self%alphas=[.1_dp,1._dp,10._dp]
    call require(size(self%alphas)>0,'RidgeClassifierCV needs candidate alphas')
    call require(all(ieee_is_finite(self%alphas)),'nonfinite RidgeClassifierCV alphas')
    call require(all(self%alphas>0),'RidgeClassifierCV alphas must be positive')
    call ridge_class_setup(self,x,y,w,xm,ym,b,s,u,v,sample_weight,class_weight)
    n=size(x,1); p=size(x,2); k=size(b,2); positive=count(w>0)
    call require(positive>=2,'RidgeClassifierCV needs at least two positive-weight samples')
    total=sum(w); projection=matmul(transpose(u),b)
    outside=b-matmul(u,projection); remainder=1-sum(u*u,dim=2)
    if(self%fit_intercept) remainder=remainder-w/total
    full_span=self%rank_==positive-merge(1,0,self%fit_intercept)
    if(full_span) then
      ! Avoid cancellation of tiny alpha terms for interpolating wide designs.
      outside=0; where(w>0) remainder=0
    end if
    remainder=max(0._dp,remainder)
    allocate(inverse(p),dual(n,k),diagonal(n),errors(n,k))
    self%cv_scores_=spread(0._dp,1,size(self%alphas))
    if(allocated(self%cv_results_)) deallocate(self%cv_results_)
    if(self%store_cv_results) allocate(self%cv_results_(n,k,size(self%alphas)))
    best=1
    do j=1,size(self%alphas)
      alpha=self%alphas(j); inverse=0
      do i=1,p
        if(s(i)>0) inverse(i)=1/(s(i)**2+alpha)
      end do
      dual=outside/alpha+matmul(u,projection*spread(inverse,2,k))
      diagonal=remainder/alpha+matmul(u*u,inverse)
      errors=0
      do i=1,n
        if(w(i)==0) cycle
        call require(diagonal(i)>0,'RidgeClassifierCV singular LOO denominator')
        errors(i,:)=(dual(i,:)/diagonal(i))**2
      end do
      score=-sum(errors)/real(n*k,dp); self%cv_scores_(j)=score
      if(self%store_cv_results) self%cv_results_(:,:,j)=errors
      if(score>self%cv_scores_(best)) best=j
    end do
    self%alpha_=self%alphas(best); self%best_score_=self%cv_scores_(best); self%alpha=self%alpha_
    call ridge_class_coefficients(self,self%alpha_,s,u,v,b,xm,ym)
  end subroutine

  function ridge_class_decision(self,x) result(scores)
    class(ridge_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    call require(allocated(self%coef_),'RidgeClassifier is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    scores=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  function ridge_class_predict(self,x) result(labels)
    class(ridge_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2) then
      labels=self%classes_(merge(2,1,scores(:,1)>0))
    else
      labels=self%classes_(maxloc(scores,dim=2))
    end if
  end function

  real(dp) function ridge_class_score(self,x,y) result(score)
    class(ridge_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    score=accuracy_score(y,self%predict(x))
  end function
end module
