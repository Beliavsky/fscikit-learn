module sklearn_model_selection
  use sklearn_weighted_linear, only: weighted_ridge_loo
  use sklearn_common
  use sklearn_resampling
  use sklearn_linear_model
  use sklearn_metrics, only: mean_squared_error
  implicit none
  private
  public :: train_test_split,shuffle_indices,kfold,stratified_kfold,time_series_split,index_split
  public :: group_kfold,leave_one_out,leave_one_group_out,predefined_split
  public :: shuffle_split,stratified_shuffle_split,group_shuffle_split
  public :: repeated_kfold,repeated_stratified_kfold,leave_p_out,leave_p_groups_out
  public :: cross_val_score,ridge_cv
  type :: kfold
    integer :: n_splits=5,random_state=0
    logical :: shuffle=.false.
  contains
    procedure :: split=>kfold_split
  end type
  type :: stratified_kfold
    integer :: n_splits=5,random_state=0
    logical :: shuffle=.false.
  contains
    procedure :: split=>stratified_split
  end type
  type :: time_series_split
    integer :: n_splits=5,test_size=0,gap=0,max_train_size=0
  contains
    procedure :: split=>time_split
  end type
  ! Dense index splitters following sklearn/model_selection/_split.py (BSD-3-Clause).
  type :: group_kfold
    integer :: n_splits=5
  contains
    procedure :: split=>group_split
  end type
  type :: leave_one_out
  contains
    procedure :: split=>loo_split
  end type
  type :: leave_one_group_out
  contains
    procedure :: split=>logo_split
  end type
  type :: predefined_split
  contains
    procedure :: split=>predefined_indices
  end type
  type, extends(ridge) :: ridge_cv
    real(dp), allocatable :: alphas(:),cv_scores_(:)
    integer :: cv=5
    real(dp) :: alpha_=0,best_score_=0
  contains
    procedure :: fit=>ridge_cv_fit
  end type
  interface cross_val_score
    module procedure cross_val_regression,cross_val_logistic
  end interface
contains
  function shuffle_indices(n,random_state) result(order)
    integer, intent(in) :: n
    integer, intent(in), optional :: random_state
    integer :: order(n),i,j,tmp
    real(dp) :: r
    call require(n>0,'cannot shuffle empty indices')
    if(present(random_state)) call seed_rng(random_state)
    order=[(i,i=1,n)]
    do i=n,2,-1
      call random_number(r); j=1+int(r*i)
      tmp=order(i); order(i)=order(j); order(j)=tmp
    end do
  end function
  subroutine train_test_split(n_samples,train,test,test_size,random_state,shuffle,stratify)
    integer, intent(in) :: n_samples
    integer, allocatable, intent(out) :: train(:),test(:)
    real(dp), intent(in), optional :: test_size
    integer, intent(in), optional :: random_state,stratify(:)
    logical, intent(in), optional :: shuffle
    integer, allocatable :: order(:),classes(:),members(:),allocations(:),counts(:),perm(:)
    real(dp), allocatable :: expected(:)
    real(dp) :: fraction
    integer :: nt,ntr,i,j,k,used,tpos,rpos,pick
    logical :: shuf
    call require(n_samples>1,'train_test_split requires at least two samples')
    fraction=0.25_dp; if(present(test_size)) fraction=test_size
    call require(fraction>0.and.fraction<1,'test_size must lie between 0 and 1')
    nt=ceiling(n_samples*fraction); ntr=n_samples-nt
    call require(ntr>0,'empty training split')
    shuf=.true.; if(present(shuffle)) shuf=shuffle
    if(present(random_state)) call seed_rng(random_state)
    if(present(stratify)) then
      call require(shuf,'stratification requires shuffling')
      call require(size(stratify)==n_samples,'stratify size mismatch')
      classes=unique_labels(stratify); k=size(classes)
      call require(nt>=k.and.ntr>=k,'split too small for every class')
      allocate(counts(k),allocations(k),expected(k))
      do i=1,k
        counts(i)=count(stratify==classes(i))
      end do
      call require(all(counts>=2),'each stratification class needs at least two samples')
      expected=real(counts,dp)*nt/n_samples
      allocations=max(1,min(counts-1,int(expected)))
      do while(sum(allocations)<nt)
        pick=0
        do i=1,k
          if(allocations(i)>=counts(i)-1) cycle
          if(pick==0) then
            pick=i
          else if(expected(i)-allocations(i)>expected(pick)-allocations(pick)) then
            pick=i
          end if
        end do
        call require(pick>0,'cannot allocate stratified test split')
        allocations(pick)=allocations(pick)+1
      end do
      do while(sum(allocations)>nt)
        pick=0
        do i=1,k
          if(allocations(i)<=1) cycle
          if(pick==0) then
            pick=i
          else if(allocations(i)-expected(i)>allocations(pick)-expected(pick)) then
            pick=i
          end if
        end do
        call require(pick>0,'cannot allocate stratified training split')
        allocations(pick)=allocations(pick)-1
      end do
      allocate(train(ntr),test(nt)); tpos=0; rpos=0
      do i=1,k
        members=pack([(j,j=1,n_samples)],stratify==classes(i)); perm=shuffle_indices(size(members))
        members=members(perm); used=allocations(i)
        test(tpos+1:tpos+used)=members(:used); tpos=tpos+used
        train(rpos+1:rpos+size(members)-used)=members(used+1:); rpos=rpos+size(members)-used
      end do
      perm=shuffle_indices(nt); test=test(perm); perm=shuffle_indices(ntr); train=train(perm)
    else
      order=[(i,i=1,n_samples)]
      if(shuf) order=shuffle_indices(n_samples)
      if(shuf) then
        test=order(:nt); train=order(nt+1:)
      else
        train=order(:ntr); test=order(ntr+1:)
      end if
    end if
  end subroutine
  function kfold_split(self,n_samples) result(folds)
    class(kfold), intent(in) :: self
    integer, intent(in) :: n_samples
    type(index_split), allocatable :: folds(:)
    integer :: i,first,last,width,n
    integer, allocatable :: order(:)
    n=self%n_splits; call require(n>=2.and.n<=n_samples,'invalid KFold n_splits')
    allocate(folds(n)); order=[(i,i=1,n_samples)]
    if(self%shuffle) order=shuffle_indices(n_samples,self%random_state)
    first=1
    do i=1,n
      width=n_samples/n+merge(1,0,i<=mod(n_samples,n)); last=first+width-1
      folds(i)%test=order(first:last); folds(i)%train=[order(:first-1),order(last+1:)]
      first=last+1
    end do
  end function
  function stratified_split(self,y) result(folds)
    class(stratified_kfold), intent(in) :: self
    integer, intent(in) :: y(:)
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: classes(:),members(:),assignment(:),permutation(:),allocation(:,:)
    integer :: c,i,j,n,k,offset,position
    n=self%n_splits; call require(n>=2.and.n<=size(y),'invalid stratified n_splits')
    classes=unique_labels(y); k=size(classes)
    allocate(folds(n),assignment(size(y)),allocation(n,k)); allocation=0
    offset=0
    do c=1,k
      call require(count(y==classes(c))>=n,'each class must have at least n_splits samples')
      do i=1,count(y==classes(c))
        j=1+mod(offset+i-1,n); allocation(j,c)=allocation(j,c)+1
      end do
      offset=offset+count(y==classes(c))
    end do
    if(self%shuffle) call seed_rng(self%random_state)
    do c=1,k
      members=pack([(i,i=1,size(y))],y==classes(c))
      if(self%shuffle) then
        permutation=shuffle_indices(size(members)); members=members(permutation)
      end if
      position=1
      do j=1,n
        assignment(members(position:position+allocation(j,c)-1))=j
        position=position+allocation(j,c)
      end do
    end do
    do i=1,n
      folds(i)%test=pack([(j,j=1,size(y))],assignment==i)
      folds(i)%train=pack([(j,j=1,size(y))],assignment/=i)
    end do
  end function
  function time_split(self,n_samples) result(folds)
    class(time_series_split), intent(in) :: self
    integer, intent(in) :: n_samples
    type(index_split), allocatable :: folds(:)
    integer :: n,width,start,finish,first,i,j
    n=self%n_splits; width=self%test_size
    call require(n>=2.and.self%gap>=0.and.self%max_train_size>=0,'invalid time series controls')
    if(width==0) width=n_samples/(n+1)
    call require(width>0.and.n_samples-n*width-self%gap>0,'not enough time series samples')
    allocate(folds(n))
    do i=1,n
      start=n_samples-(n-i+1)*width+1; finish=start-self%gap-1; first=1
      if(self%max_train_size>0) first=max(1,finish-self%max_train_size+1)
      folds(i)%train=[(j,j=first,finish)]; folds(i)%test=[(j,j=start,start+width-1)]
    end do
  end function
  function group_split(self,groups) result(folds)
    class(group_kfold), intent(in) :: self
    integer, intent(in) :: groups(:)
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: labels(:),counts(:),order(:),loads(:),assignment(:),ids(:)
    integer :: i,j,k,c,key,n
    call require(size(groups)>0,'GroupKFold groups must be nonempty')
    labels=unique_labels(groups); n=self%n_splits; k=size(labels)
    call require(n>=2.and.n<=k,'GroupKFold requires 2 <= n_splits <= number of groups')
    allocate(counts(k),order(k),loads(n),assignment(size(groups)))
    do i=1,k
      counts(i)=count(groups==labels(i)); order(i)=i
    end do
    ! Largest group first; equal counts use descending numeric group labels.
    ! NumPy quicksort does not guarantee this tie ordering on every version.
    do i=2,k
      key=order(i); j=i-1
      do while(j>=1)
        c=order(j)
        if(counts(c)>counts(key)) exit
        if(counts(c)==counts(key).and.c>key) exit
        order(j+1)=c; j=j-1
      end do
      order(j+1)=key
    end do
    loads=0
    do i=1,k
      c=order(i); j=minloc(loads,dim=1)
      where(groups==labels(c)) assignment=j
      loads(j)=loads(j)+counts(c)
    end do
    allocate(folds(n)); ids=[(i,i=1,size(groups))]
    do i=1,n
      folds(i)%test=pack(ids,assignment==i); folds(i)%train=pack(ids,assignment/=i)
    end do
  end function
  function loo_split(self,n_samples) result(folds)
    class(leave_one_out), intent(in) :: self
    integer, intent(in) :: n_samples
    type(index_split), allocatable :: folds(:)
    integer :: i,j
    call require(n_samples>=2,'LeaveOneOut requires at least two samples')
    allocate(folds(n_samples))
    do i=1,n_samples
      folds(i)%test=[i]; folds(i)%train=[(j,j=1,i-1),(j,j=i+1,n_samples)]
    end do
  end function
  function logo_split(self,groups) result(folds)
    class(leave_one_group_out), intent(in) :: self
    integer, intent(in) :: groups(:)
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: labels(:),ids(:)
    integer :: i
    call require(size(groups)>0,'LeaveOneGroupOut groups must be nonempty')
    labels=unique_labels(groups)
    call require(size(labels)>=2,'LeaveOneGroupOut requires at least two groups')
    allocate(folds(size(labels))); ids=[(i,i=1,size(groups))]
    do i=1,size(labels)
      folds(i)%test=pack(ids,groups==labels(i)); folds(i)%train=pack(ids,groups/=labels(i))
    end do
  end function
  function predefined_indices(self,test_fold) result(folds)
    class(predefined_split), intent(in) :: self
    integer, intent(in) :: test_fold(:)
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: labels(:),ids(:)
    integer :: i
    ! -1 is the only special label. Other integers, including negatives, are folds.
    if(size(test_fold)==0) then
      allocate(folds(0)); return
    end if
    labels=unique_labels(test_fold); labels=pack(labels,labels/=-1)
    allocate(folds(size(labels))); ids=[(i,i=1,size(test_fold))]
    do i=1,size(labels)
      folds(i)%test=pack(ids,test_fold==labels(i)); folds(i)%train=pack(ids,test_fold/=labels(i))
    end do
  end function
  function cross_val_regression(estimator,x,y,cv) result(scores)
    class(linear_regression), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in), optional :: cv
    real(dp), allocatable :: scores(:)
    class(linear_regression), allocatable :: model
    type(kfold) :: splitter
    type(index_split), allocatable :: folds(:)
    integer :: i
    call check_xy(x,size(y)); if(present(cv)) splitter%n_splits=cv
    folds=splitter%split(size(y)); allocate(scores(size(folds)))
    allocate(model,source=estimator)
    do i=1,size(folds)
      call model%fit(x(folds(i)%train,:),y(folds(i)%train))
      scores(i)=model%score(x(folds(i)%test,:),y(folds(i)%test))
    end do
  end function
  function cross_val_logistic(estimator,x,y,cv) result(scores)
    type(logistic_regression), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, intent(in), optional :: cv
    real(dp), allocatable :: scores(:)
    type(logistic_regression) :: model
    type(stratified_kfold) :: splitter
    type(index_split), allocatable :: folds(:)
    integer :: i
    call check_xy(x,size(y)); if(present(cv)) splitter%n_splits=cv
    folds=splitter%split(y); allocate(scores(size(folds))); model=estimator
    do i=1,size(folds)
      call model%fit(x(folds(i)%train,:),y(folds(i)%train))
      scores(i)=model%score(x(folds(i)%test,:),y(folds(i)%test))
    end do
  end function
  subroutine ridge_cv_fit(self,x,y)
    class(ridge_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(ridge) :: model
    type(kfold) :: splitter
    type(index_split), allocatable :: folds(:)
    real(dp) :: score
    integer :: i,j,best
    call check_xy(x,size(y))
    if(.not.allocated(self%alphas)) self%alphas=[0.1_dp,1._dp,10._dp]
    if(self%cv==0) then
      call weighted_ridge_loo(self,x,y,self%alphas,self%alpha_,self%best_score_,self%cv_scores_)
      self%alpha=self%alpha_; return
    end if
    call require(size(self%alphas)>0.and.all(self%alphas>=0),'invalid ridge CV alphas')
    splitter%n_splits=self%cv; folds=splitter%split(size(y)); model%fit_intercept=self%fit_intercept
    self%cv_scores_=self%alphas*0
    do i=1,size(self%alphas)
      model%alpha=self%alphas(i); score=0
      do j=1,size(folds)
        call model%fit(x(folds(j)%train,:),y(folds(j)%train))
        score=score-mean_squared_error(y(folds(j)%test),model%predict(x(folds(j)%test,:)))
      end do
      self%cv_scores_(i)=score/size(folds)
    end do
    best=maxloc(self%cv_scores_,dim=1); self%alpha_=self%alphas(best)
    self%best_score_=self%cv_scores_(best); self%alpha=self%alpha_
    call self%ridge%fit(x,y)
  end subroutine
end module
