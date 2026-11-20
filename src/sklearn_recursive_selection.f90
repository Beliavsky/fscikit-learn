! Native adaptation of sklearn 1.6.1 feature_selection/_rfe.py (BSD-3-Clause).
module sklearn_recursive_selection
  use sklearn_common
  use sklearn_linear_model, only: linear_regression,logistic_regression
  use sklearn_model_selection, only: kfold,index_split
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: rfe,rfecv,rfecv_results
  type :: rfecv_results
    integer, allocatable :: n_features(:)
    real(dp), allocatable :: mean_test_score(:),std_test_score(:),split_test_score(:,:)
  end type
  type :: rfe
    integer :: n_features_to_select=0,n_features_=0,n_features_in_=0
    real(dp) :: step=1
    logical, allocatable :: support_(:)
    integer, allocatable :: ranking_(:)
    class(linear_regression), allocatable :: regression_estimator_
    type(logistic_regression), allocatable :: classifier_estimator_
  contains
    procedure :: fit_regression=>rfe_reg_fit
    procedure :: fit_classification=>rfe_class_fit
    generic :: fit=>fit_regression,fit_classification
    procedure :: fit_transform_regression=>rfe_reg_fit_transform
    procedure :: fit_transform_classification=>rfe_class_fit_transform
    generic :: fit_transform=>fit_transform_regression,fit_transform_classification
    procedure :: transform=>rfe_transform
    procedure :: inverse_transform=>rfe_inverse
    procedure :: get_support=>rfe_support
    procedure :: predict=>rfe_predict
    procedure :: predict_classes=>rfe_predict_classes
    procedure :: predict_proba=>rfe_proba
    procedure :: decision_function=>rfe_decision
    procedure :: score_regression=>rfe_reg_score
    procedure :: score_classification=>rfe_class_score
    generic :: score=>score_regression,score_classification
  end type
  type, extends(rfe) :: rfecv
    integer :: min_features_to_select=1,cv=5
    type(rfecv_results) :: cv_results_
  contains
    procedure :: fit_regression=>rfecv_reg_fit
    procedure :: fit_classification=>rfecv_class_fit
  end type
contains
  integer function elimination_step(self,p) result(width)
    class(rfe), intent(in) :: self
    integer, intent(in) :: p
    call require(ieee_is_finite(self%step),'nonfinite RFE step')
    call require(self%step>0,'RFE step must be positive')
    if(self%step<1) then
      width=max(1,int(self%step*p))
    else
      width=int(min(self%step,real(p,dp)))
    end if
  end function
  integer function selection_target(self,p) result(target)
    class(rfe), intent(in) :: self
    integer, intent(in) :: p
    call require(self%n_features_to_select>=0,'RFE feature count must be nonnegative')
    target=self%n_features_to_select
    if(target==0) target=p/2
    target=min(target,p)
  end function
  subroutine rfe_reg_fit(self,x,y,estimator)
    class(rfe), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(linear_regression), intent(in) :: estimator
    call run_elimination(self,x,selection_target(self,size(x,2)),yr=y,reg=estimator)
  end subroutine
  subroutine rfe_class_fit(self,x,y,estimator)
    class(rfe), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(logistic_regression), intent(in) :: estimator
    call run_elimination(self,x,selection_target(self,size(x,2)),yc=y,clf=estimator)
  end subroutine
  subroutine run_elimination(self,x,target,yr,yc,reg,clf,test_x,test_yr,test_yc,path,counts)
    class(rfe), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: target
    real(dp), intent(in), optional :: yr(:),test_x(:,:),test_yr(:)
    integer, intent(in), optional :: yc(:),test_yc(:)
    class(linear_regression), intent(in), optional :: reg
    type(logistic_regression), intent(in), optional :: clf
    real(dp), allocatable, intent(out), optional :: path(:)
    integer, allocatable, intent(out), optional :: counts(:)
    real(dp), allocatable :: importance(:)
    integer, allocatable :: ids(:),features(:),order(:)
    integer :: p,width,nstage,stage,drop,j
    call check_x(x); p=size(x,2)
    call require(p>=2,'RFE requires at least two input features')
    width=elimination_step(self,p)
    if(present(yr)) then
      call check_xy(x,size(yr)); call check_x(reshape(yr,[size(yr),1]))
    else
      call check_xy(x,size(yc))
    end if
    if(allocated(self%regression_estimator_)) deallocate(self%regression_estimator_)
    if(allocated(self%classifier_estimator_)) deallocate(self%classifier_estimator_)
    self%n_features_in_=p; self%support_=[( .true.,j=1,p)]; self%ranking_=[(1,j=1,p)]
    ids=[(j,j=1,p)]; nstage=(p-target+width-1)/width+1
    if(present(path)) allocate(path(nstage))
    if(present(counts)) allocate(counts(nstage))
    do stage=1,nstage
      features=pack(ids,self%support_)
      if(present(reg)) then
        if(allocated(self%regression_estimator_)) deallocate(self%regression_estimator_)
        allocate(self%regression_estimator_,source=reg)
        call self%regression_estimator_%fit(x(:,features),yr)
        importance=self%regression_estimator_%coef_**2
        if(present(path)) path(stage)=self%regression_estimator_%score(test_x(:,features),test_yr)
      else
        if(allocated(self%classifier_estimator_)) deallocate(self%classifier_estimator_)
        allocate(self%classifier_estimator_,source=clf)
        call self%classifier_estimator_%fit(x(:,features),yc)
        importance=sum(self%classifier_estimator_%coef_**2,dim=1)
        if(present(path)) path(stage)=self%classifier_estimator_%score(test_x(:,features),test_yc)
      end if
      call require(all(ieee_is_finite(importance)),'nonfinite RFE coefficient importance')
      if(present(counts)) counts(stage)=size(features)
      if(size(features)<=target) exit
      ! Stable ascending importance: equal importances eliminate earlier columns first.
      order=argsort(importance); drop=min(width,size(features)-target)
      self%support_(features(order(:drop)))=.false.
      where(.not.self%support_) self%ranking_=self%ranking_+1
    end do
    self%n_features_=count(self%support_)
  end subroutine
  subroutine rfecv_reg_fit(self,x,y,estimator)
    class(rfecv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(linear_regression), intent(in) :: estimator
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%cv>=2.and.self%cv<=size(y)/2,'RFECV regression needs at least two test rows per fold')
    call run_cv(self,x,yr=y,reg=estimator)
  end subroutine
  subroutine rfecv_class_fit(self,x,y,estimator)
    class(rfecv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(logistic_regression), intent(in) :: estimator
    call check_xy(x,size(y))
    call require(self%cv>=2.and.self%cv<=size(y),'invalid RFECV classification cv')
    call run_cv(self,x,yc=y,clf=estimator)
  end subroutine
  subroutine run_cv(self,x,yr,yc,reg,clf)
    class(rfecv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: yr(:)
    integer, intent(in), optional :: yc(:)
    class(linear_regression), intent(in), optional :: reg
    type(logistic_regression), intent(in), optional :: clf
    type(kfold) :: splitter
    type(index_split), allocatable :: folds(:)
    type(rfe) :: fold_model
    real(dp), allocatable :: path(:),scores(:,:)
    integer, allocatable :: counts(:)
    integer :: p,width,target,stages,i,j,best
    p=size(x,2); call require(p>=2,'RFE requires at least two input features')
    call require(self%min_features_to_select>=1,'RFECV minimum feature count must be positive')
    width=elimination_step(self,p); target=min(p,self%min_features_to_select)
    stages=(p-target+width-1)/width+1; allocate(scores(stages,self%cv))
    if(present(reg)) then
      splitter%n_splits=self%cv; folds=splitter%split(size(yr))
    else
      folds=classification_folds(yc,self%cv)
    end if
    fold_model%step=self%step
    do i=1,self%cv
      if(present(reg)) then
        call run_elimination(fold_model,x(folds(i)%train,:),target,yr=yr(folds(i)%train),reg=reg, &
          test_x=x(folds(i)%test,:),test_yr=yr(folds(i)%test),path=path,counts=counts)
      else
        call run_elimination(fold_model,x(folds(i)%train,:),target,yc=yc(folds(i)%train),clf=clf, &
          test_x=x(folds(i)%test,:),test_yc=yc(folds(i)%test),path=path,counts=counts)
      end if
      call require(all(ieee_is_finite(path)),'nonfinite RFECV score')
      scores(:,i)=path(stages:1:-1)
    end do
    self%cv_results_%n_features=counts(stages:1:-1)
    self%cv_results_%split_test_score=scores
    self%cv_results_%mean_test_score=sum(scores,dim=2)/self%cv
    self%cv_results_%std_test_score=[(sqrt(sum((scores(j,:)-self%cv_results_%mean_test_score(j))**2)/self%cv), &
      j=1,stages)]
    ! Ascending feature counts give the smallest subset on exact score ties.
    best=maxloc(sum(scores,dim=2),dim=1); target=self%cv_results_%n_features(best)
    call run_elimination(self,x,target,yr,yc,reg,clf)
  end subroutine
  function classification_folds(y,n) result(folds)
    integer, intent(in) :: y(:),n
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: labels(:),first(:),order(:),assignment(:),members(:),ids(:),counts(:)
    integer :: k,c,i,j,offset,pos,width
    labels=unique_labels(y); k=size(labels)
    call require(k>=2,'RFECV classification requires at least two classes')
    allocate(first(k),counts(k),assignment(size(y)))
    do c=1,k
      first(c)=findloc(y,labels(c),dim=1); counts(c)=count(y==labels(c))
    end do
    call require(minval(counts)>=2,'RFECV classification requires at least two samples per class')
    call require(maxval(counts)>=n,'RFECV cv exceeds every class count')
    ! sklearn StratifiedKFold encodes classes by first appearance, not numeric label.
    order=argsort(real(first,dp)); ids=[(i,i=1,size(y))]; offset=0
    do i=1,k
      c=order(i); members=pack(ids,y==labels(c)); pos=1
      do j=1,n
        width=(offset+counts(c)+n-j)/n-(offset+n-j)/n
        assignment(members(pos:pos+width-1))=j; pos=pos+width
      end do
      offset=offset+counts(c)
    end do
    allocate(folds(n))
    do i=1,n
      folds(i)%train=pack(ids,assignment/=i); folds(i)%test=pack(ids,assignment==i)
    end do
  end function
  function rfe_reg_fit_transform(self,x,y,estimator) result(z)
    class(rfe), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(linear_regression), intent(in) :: estimator
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y,estimator); z=self%transform(x)
  end function
  function rfe_class_fit_transform(self,x,y,estimator) result(z)
    class(rfe), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(logistic_regression), intent(in) :: estimator
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y,estimator); z=self%transform(x)
  end function
  function rfe_transform(self,x) result(z)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: j
    call require(allocated(self%support_),'RFE is not fitted'); call check_x(x)
    call require(size(x,2)==size(self%support_),'RFE feature count mismatch')
    z=x(:,pack([(j,j=1,size(x,2))],self%support_))
  end function
  function rfe_inverse(self,x) result(z)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: j
    call require(allocated(self%support_),'RFE is not fitted'); call check_x(x)
    call require(size(x,2)==count(self%support_),'RFE inverse feature count mismatch')
    allocate(z(size(x,1),size(self%support_))); z=0
    z(:,pack([(j,j=1,size(self%support_))],self%support_))=x
  end function
  function rfe_support(self) result(mask)
    class(rfe), intent(in) :: self
    logical, allocatable :: mask(:)
    call require(allocated(self%support_),'RFE is not fitted'); mask=self%support_
  end function
  function rfe_predict(self,x) result(y)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%regression_estimator_),'RFE has no fitted regression estimator')
    y=self%regression_estimator_%predict(self%transform(x))
  end function
  function rfe_predict_classes(self,x) result(y)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    call require(allocated(self%classifier_estimator_),'RFE has no fitted classifier')
    y=self%classifier_estimator_%predict(self%transform(x))
  end function
  function rfe_proba(self,x) result(prob)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    call require(allocated(self%classifier_estimator_),'RFE has no fitted classifier')
    prob=self%classifier_estimator_%predict_proba(self%transform(x))
  end function
  function rfe_decision(self,x) result(z)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%classifier_estimator_),'RFE has no fitted classifier')
    z=self%classifier_estimator_%decision_function(self%transform(x))
  end function
  real(dp) function rfe_reg_score(self,x,y) result(s)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(allocated(self%regression_estimator_),'RFE has no fitted regression estimator')
    s=self%regression_estimator_%score(self%transform(x),y)
  end function
  real(dp) function rfe_class_score(self,x,y) result(s)
    class(rfe), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call check_xy(x,size(y))
    call require(allocated(self%classifier_estimator_),'RFE has no fitted classifier')
    s=self%classifier_estimator_%score(self%transform(x),y)
  end function
end module
