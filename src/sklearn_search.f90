! Dense GridSearchCV/RandomizedSearchCV contracts: sklearn 1.6.1
! model_selection/_search.py, _validation.py and _split.py (BSD-3-Clause).
! Native typed interfaces are original MIT contributions.
module sklearn_search
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_common, only: dp,require,check_xy,unique_labels,argsort
  use sklearn_parameter_space
  use sklearn_search_adapters
  use sklearn_model_selection, only: index_split,kfold
  use sklearn_ieee_math, only: ieee_mean,ieee_difference,ieee_product
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: grid_search_cv,randomized_search_cv,search_cv_results,search_scorer
  public :: parameter_grid,parameter_axis,parameter_value,parameter_item,parameter_set,param
  public :: enumerate_grids,sample_parameters
  type :: search_cv_results
    type(parameter_set), allocatable :: params(:)
    character(64), allocatable :: metric_names(:)
    real(dp), allocatable :: split_test_score(:,:,:),mean_test_score(:,:),std_test_score(:,:)
    real(dp), allocatable :: split_train_score(:,:,:),mean_train_score(:,:),std_train_score(:,:)
    integer, allocatable :: rank_test_score(:,:)
    real(dp), allocatable :: fit_time(:,:),score_time(:,:),mean_fit_time(:),std_fit_time(:), &
      mean_score_time(:),std_score_time(:)
  end type
  abstract interface
    real(dp) function search_score_function(model,x,yr,yi) result(value)
      import dp
      class(*), intent(in) :: model
      real(dp), intent(in) :: x(:,:)
      real(dp), optional, intent(in) :: yr(:)
      integer, optional, intent(in) :: yi(:)
    end function
    subroutine search_fit_function(model,x,yr,yi,sample_weight)
      import dp
      class(*), intent(inout) :: model
      real(dp), intent(in) :: x(:,:)
      real(dp), optional, intent(in) :: yr(:),sample_weight(:)
      integer, optional, intent(in) :: yi(:)
    end subroutine
    subroutine search_parameter_function(model,parameters)
      import parameter_set
      class(*), intent(inout) :: model
      type(parameter_set), intent(in) :: parameters
    end subroutine
    integer function search_refit_function(results) result(index)
      import search_cv_results
      type(search_cv_results), intent(in) :: results
    end function
  end interface
  type :: search_scorer
    character(64) :: name='score',scoring='auto'
    procedure(search_score_function), pointer, nopass :: func=>null()
  end type
  type :: grid_search_cv
    class(*), allocatable :: estimator,best_estimator_
    type(parameter_grid), allocatable :: param_grid(:)
    type(index_split), allocatable :: cv_splits(:),folds_(:)
    type(search_scorer), allocatable :: scorers(:),scorers_(:)
    character(64) :: scoring='auto',refit_metric=''
    character(16) :: task='auto',error_score='raise'
    integer :: cv=5,n_splits_=0,n_features_in_=0,best_index_=0,refit_metric_=0
    logical :: refit=.true.,return_train_score=.false.,multimetric_=.false.,best_score_available_=.false.
    real(dp) :: best_score_=0,refit_time_=0
    type(parameter_set) :: best_params_
    type(search_cv_results) :: cv_results_
    integer, allocatable :: classes_(:)
    procedure(search_score_function), pointer, nopass :: scorer_func=>null()
    procedure(search_parameter_function), pointer, nopass :: set_params_func=>null()
    procedure(search_fit_function), pointer, nopass :: fit_func=>null()
    procedure(search_refit_function), pointer, nopass :: refit_func=>null()
    logical, private :: fitted_=.false.,classifier_=.false.
  contains
    procedure :: fit_real,fit_integer
    generic :: fit=>fit_real,fit_integer
    procedure :: predict,predict_class,predict_proba,decision_function
    procedure :: score_real,score_integer
    generic :: score=>score_real,score_integer
  end type
  type, extends(grid_search_cv) :: randomized_search_cv
    integer :: n_iter=10,random_state=0
  end type
contains
  function integral_labels(y) result(labels)
    real(dp), intent(in) :: y(:)
    integer, allocatable :: labels(:)
    call require(all(ieee_is_finite(y)),'search classifier targets must be finite')
    call require(all(y>=-real(huge(1),dp)-1).and.all(y<=real(huge(1),dp)), &
      'search classifier label exceeds integer range')
    call require(all(modulo(y,1._dp)==0),'search classifier targets must be integral')
    labels=int(y)
  end function
  logical function classifier_task(self) result(classifier)
    class(grid_search_cv), intent(in) :: self
    call require(allocated(self%estimator),'search estimator is unallocated')
    select case(self%task)
    case('classification');classifier=.true.
    case('regression');classifier=.false.
    case('auto');classifier=search_classifier(self%estimator)
    case default;call require(.false.,'search task must be auto, classification or regression')
    end select
  end function
  subroutine fit_real(self,x,y,sample_weight)
    class(grid_search_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: labels(:)
    call check_xy(x,size(y));call require(all(ieee_is_finite(y)),'search targets must be finite')
    if(classifier_task(self)) then
      labels=integral_labels(y);call fit_core(self,x,yi=labels,sample_weight=sample_weight)
    else
      call fit_core(self,x,yr=y,sample_weight=sample_weight)
    end if
  end subroutine
  subroutine fit_integer(self,x,y,sample_weight)
    class(grid_search_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call check_xy(x,size(y))
    if(classifier_task(self)) then
      call fit_core(self,x,yi=y,sample_weight=sample_weight)
    else
      call fit_core(self,x,yr=real(y,dp),sample_weight=sample_weight)
    end if
  end subroutine
  function stratified_folds(y,n) result(folds)
    integer, intent(in) :: y(:),n
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: classes(:),members(:),first(:),order(:),counts(:),assignment(:)
    integer :: c,j,i,offset,width,start,c_block
    call require(n>=2.and.n<=size(y),'search invalid stratified cv')
    classes=unique_labels(y);allocate(first(size(classes)),counts(size(classes)),assignment(size(y)))
    do c=1,size(classes)
      first(c)=findloc(y,classes(c),dim=1);counts(c)=count(y==classes(c))
    end do
    call require(.not.all(counts<n),'search cv exceeds every class count')
    order=argsort(real(first,dp));offset=0
    do i=1,size(classes)
      c=order(i);members=pack([(j,j=1,size(y))],y==classes(c));start=1
      do j=1,n
        width=0
        width=count(modulo([(offset+c_block-1,c_block=1,size(members))],n)==j-1)
        assignment(members(start:start+width-1))=j;start=start+width
      end do
      offset=offset+size(members)
    end do
    allocate(folds(n))
    do i=1,n
      folds(i)%train=pack([(j,j=1,size(y))],assignment/=i)
      folds(i)%test=pack([(j,j=1,size(y))],assignment==i)
    end do
  end function
  subroutine validate_folds(folds,n)
    type(index_split), intent(in) :: folds(:)
    integer, intent(in) :: n
    integer :: i
    call require(size(folds)>0,'search requires at least one CV split')
    do i=1,size(folds)
      call require(allocated(folds(i)%train).and.allocated(folds(i)%test),'search CV indices are unallocated')
      call require(size(folds(i)%train)>0.and.size(folds(i)%test)>0,'search CV splits must be nonempty')
      call require(all(folds(i)%train>=1).and.all(folds(i)%train<=n),'search train index out of bounds')
      call require(all(folds(i)%test>=1).and.all(folds(i)%test<=n),'search test index out of bounds')
    end do
  end subroutine
  subroutine configure_scorers(self)
    class(grid_search_cv), intent(inout) :: self
    integer :: i,j
    if(allocated(self%scorers_)) deallocate(self%scorers_)
    self%multimetric_=allocated(self%scorers)
    if(self%multimetric_) then
      call require(.not.associated(self%scorer_func),'search choose scorer_func or scorers')
      call require(size(self%scorers)>0,'search scorers must be nonempty');self%scorers_=self%scorers
    else
      allocate(self%scorers_(1));self%scorers_(1)%scoring=self%scoring
      self%scorers_(1)%func=>self%scorer_func
    end if
    self%refit_metric_=0
    do i=1,size(self%scorers_)
      call require(len_trim(self%scorers_(i)%name)>0,'search scorer name must be nonempty')
      do j=1,i-1
        call require(self%scorers_(i)%name/=self%scorers_(j)%name,'search scorer names must be unique')
      end do
      if(self%scorers_(i)%name==self%refit_metric) self%refit_metric_=i
    end do
    if(.not.self%multimetric_) self%refit_metric_=1
    if(self%refit.and..not.associated(self%refit_func)) &
      call require(self%refit_metric_>0,'search multimetric refit requires a valid refit_metric')
  end subroutine
  real(dp) function evaluate(scorer,model,x,yr,yi) result(value)
    type(search_scorer), intent(in) :: scorer
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    if(associated(scorer%func)) then
      value=scorer%func(model,x,yr,yi)
    else
      value=native_search_score(model,x,scorer%scoring,yr,yi)
    end if
  end function
  subroutine train_candidate(self,model,parameters,x,yr,yi,weights)
    class(grid_search_cv), intent(in) :: self
    class(*), allocatable, intent(out) :: model
    type(parameter_set), intent(in) :: parameters
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),weights(:)
    integer, optional, intent(in) :: yi(:)
    allocate(model,source=self%estimator)
    if(associated(self%set_params_func)) then
      call self%set_params_func(model,parameters)
    else
      call set_search_params(model,parameters)
    end if
    if(associated(self%fit_func)) then
      call self%fit_func(model,x,yr,yi,weights)
    else
      call fit_search_native(model,x,yr,yi,weights)
    end if
  end subroutine
  real(dp) function elapsed_clock(start) result(value)
    integer(int64), intent(in) :: start
    integer(int64) :: now,rate
    call system_clock(now,rate);value=max(0._dp,real(now-start,dp)/real(rate,dp))
  end function
  subroutine fit_core(self,x,yr,yi,sample_weight)
    class(grid_search_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:)
    type(kfold) :: splitter
    type(parameter_set), allocatable :: candidates(:)
    type(search_cv_results) :: empty
    class(*), allocatable :: model
    real(dp), allocatable :: train_y(:),test_y(:),weights(:),test_score(:),train_score(:)
    integer, allocatable :: train_labels(:),test_labels(:)
    integer :: i,j,k,nc,nf,nm,best
    integer(int64) :: start
    call require(self%error_score=='raise','native search supports only error_score=raise')
    call require(.not.associated(self%refit_func).or.self%refit,'search refit callback requires refit=true')
    call require(allocated(self%param_grid),'search param_grid is unallocated')
    self%fitted_=.false.;self%classifier_=present(yi);self%best_index_=0;self%refit_time_=0
    self%best_score_=ieee_value(0._dp,ieee_quiet_nan);self%best_score_available_=.false.
    self%best_params_=parameter_set();self%cv_results_=empty
    if(allocated(self%best_estimator_)) deallocate(self%best_estimator_)
    if(allocated(self%classes_)) deallocate(self%classes_)
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(x,1),'search fit weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'search fit weights must be finite')
    end if
    select type(self)
    class is(randomized_search_cv)
      candidates=sample_parameters(self%param_grid,self%n_iter,self%random_state)
    class default
      candidates=enumerate_grids(self%param_grid)
    end select
    call require(size(candidates)>0,'search requires at least one candidate')
    if(allocated(self%cv_splits)) then
      self%folds_=self%cv_splits
    else if(present(yi)) then
      self%folds_=stratified_folds(yi,self%cv)
    else
      splitter%n_splits=self%cv;self%folds_=splitter%split(size(x,1))
    end if
    call validate_folds(self%folds_,size(x,1));call configure_scorers(self)
    nc=size(candidates);nf=size(self%folds_);nm=size(self%scorers_)
    self%n_splits_=nf;self%n_features_in_=size(x,2);self%cv_results_%params=candidates
    allocate(self%cv_results_%metric_names(nm));self%cv_results_%metric_names=self%scorers_%name
    allocate(self%cv_results_%split_test_score(nc,nf,nm),self%cv_results_%fit_time(nc,nf), &
      self%cv_results_%score_time(nc,nf))
    if(self%return_train_score) allocate(self%cv_results_%split_train_score(nc,nf,nm))
    do i=1,nc
      do j=1,nf
        associate(fold=>self%folds_(j))
          if(present(yr)) then
            train_y=yr(fold%train);test_y=yr(fold%test)
          end if
          if(present(yi)) then
            train_labels=yi(fold%train);test_labels=yi(fold%test)
          end if
          if(present(sample_weight)) weights=sample_weight(fold%train)
          call system_clock(start)
          call train_candidate(self,model,candidates(i),x(fold%train,:),train_y,train_labels,weights)
          self%cv_results_%fit_time(i,j)=elapsed_clock(start)
          call system_clock(start)
          do k=1,nm
            self%cv_results_%split_test_score(i,j,k)=evaluate(self%scorers_(k),model,x(fold%test,:),test_y,test_labels)
          end do
          self%cv_results_%score_time(i,j)=elapsed_clock(start)
          if(self%return_train_score) then
            do k=1,nm
              self%cv_results_%split_train_score(i,j,k)=evaluate(self%scorers_(k),model,x(fold%train,:), &
                train_y,train_labels)
            end do
          end if
        end associate
      end do
    end do
    call format_results(self%cv_results_,self%return_train_score)
    if(associated(self%refit_func)) then
      best=self%refit_func(self%cv_results_)
      call require(best>=1.and.best<=nc,'search refit callback index out of bounds')
    else if(self%refit_metric_>0.and.(self%refit.or..not.self%multimetric_)) then
      best=minloc(self%cv_results_%rank_test_score(:,self%refit_metric_),dim=1)
      self%best_score_available_=.true.;self%best_score_=self%cv_results_%mean_test_score(best,self%refit_metric_)
    else
      best=0
    end if
    if(best>0) then
      self%best_index_=best;self%best_params_=candidates(best)
    end if
    if(self%refit) then
      call require(best>0,'search refit did not select a candidate')
      call system_clock(start)
      call train_candidate(self,model,candidates(best),x,yr,yi,sample_weight)
      call move_alloc(model,self%best_estimator_)
      self%refit_time_=elapsed_clock(start)
      if(present(yi)) self%classes_=unique_labels(yi)
    end if
    self%fitted_=.true.
  end subroutine
  subroutine mean_std(values,mean,std)
    real(dp), intent(in) :: values(:)
    real(dp), intent(out) :: mean,std
    real(dp) :: deviations(size(values)),d
    integer :: i
    mean=ieee_mean(values)
    do i=1,size(values)
      d=ieee_difference(values(i),mean);deviations(i)=ieee_product(d,d)
    end do
    std=sqrt(ieee_mean(deviations))
  end subroutine
  function ranks(scores) result(rank)
    real(dp), intent(in) :: scores(:)
    integer :: rank(size(scores)),i
    real(dp) :: mapped(size(scores)),worst
    rank=1;if(all(ieee_is_nan(scores))) return
    worst=minval(scores,mask=.not.ieee_is_nan(scores));worst=ieee_difference(worst,1._dp)
    do i=1,size(scores)
      if(ieee_is_nan(scores(i))) then
        mapped(i)=worst
      else if(.not.ieee_is_finite(scores(i))) then
        mapped(i)=sign(huge(1._dp),scores(i))
      else
        mapped(i)=scores(i)
      end if
    end do
    do i=1,size(scores)
      rank(i)=1+count(mapped>mapped(i))
    end do
  end function
  subroutine format_results(results,return_train)
    type(search_cv_results), intent(inout) :: results
    logical, intent(in) :: return_train
    integer :: i,j,n,m
    n=size(results%params);m=size(results%metric_names)
    allocate(results%mean_test_score(n,m),results%std_test_score(n,m),results%rank_test_score(n,m), &
      results%mean_fit_time(n),results%std_fit_time(n),results%mean_score_time(n),results%std_score_time(n))
    if(return_train) allocate(results%mean_train_score(n,m),results%std_train_score(n,m))
    do i=1,n
      call mean_std(results%fit_time(i,:),results%mean_fit_time(i),results%std_fit_time(i))
      call mean_std(results%score_time(i,:),results%mean_score_time(i),results%std_score_time(i))
      do j=1,m
        call mean_std(results%split_test_score(i,:,j),results%mean_test_score(i,j),results%std_test_score(i,j))
        if(return_train) call mean_std(results%split_train_score(i,:,j), &
          results%mean_train_score(i,j),results%std_train_score(i,j))
      end do
    end do
    do j=1,m
      results%rank_test_score(:,j)=ranks(results%mean_test_score(:,j))
    end do
  end subroutine
  subroutine fitted_for_prediction(self)
    class(grid_search_cv), intent(in) :: self
    call require(self%fitted_,'search is not fitted')
    call require(self%refit.and.allocated(self%best_estimator_),'search prediction requires refit')
  end subroutine
  function predict(self,x) result(y)
    class(grid_search_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call fitted_for_prediction(self);call require(.not.self%classifier_,'search predict requires a regressor')
    y=search_regression(self%best_estimator_,x)
  end function
  function predict_class(self,x) result(y)
    class(grid_search_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    call fitted_for_prediction(self);call require(self%classifier_,'search predict_class requires a classifier')
    y=search_labels(self%best_estimator_,x)
  end function
  function predict_proba(self,x) result(y)
    class(grid_search_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call fitted_for_prediction(self);call require(self%classifier_,'search predict_proba requires a classifier')
    y=search_probabilities(self%best_estimator_,x)
  end function
  function decision_function(self,x) result(y)
    class(grid_search_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call fitted_for_prediction(self);call require(self%classifier_,'search decision_function requires a classifier')
    y=search_decisions(self%best_estimator_,x)
  end function
  real(dp) function score_real(self,x,y) result(value)
    class(grid_search_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer, allocatable :: labels(:)
    call fitted_for_prediction(self)
    call require(self%refit_metric_>0,'search score needs a selected refit metric')
    if(self%classifier_) then
      labels=integral_labels(y);value=evaluate(self%scorers_(self%refit_metric_),self%best_estimator_,x,yi=labels)
    else
      value=evaluate(self%scorers_(self%refit_metric_),self%best_estimator_,x,yr=y)
    end if
  end function
  real(dp) function score_integer(self,x,y) result(value)
    class(grid_search_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=self%score_real(x,real(y,dp))
  end function
end module
