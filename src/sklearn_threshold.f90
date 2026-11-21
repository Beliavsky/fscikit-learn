! Binary threshold wrappers adapted from sklearn 1.6.1
! model_selection/_classification_threshold.py and metrics/_scorer.py.
! Retained BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md.
module sklearn_threshold
  use sklearn_common, only: dp,require,check_xy,check_x,unique_labels
  use sklearn_base_dispatch, only: base_supported,base_fitted,base_proba,base_decision,base_weighted, &
    base_classes,train_classifier,probabilities,decisions
  use sklearn_resampling, only: index_split,stratified_shuffle_split
  use sklearn_composite_scores, only: score_weights
  use sklearn_ieee_math, only: ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: fixed_threshold_classifier,tuned_threshold_classifier_cv
  type, abstract :: threshold_classifier_base
    character(20) :: response_method='auto'
    integer, allocatable :: pos_label,classes_(:)
    integer :: n_features_in_=0
    logical :: prefit=.false.
    class(*), allocatable :: estimator_
  contains
    procedure(threshold_prediction), deferred :: predict
    procedure :: predict_proba=>threshold_proba
    procedure :: decision_function=>threshold_decision
    procedure :: score=>threshold_score
  end type
  type, extends(threshold_classifier_base) :: fixed_threshold_classifier
    real(dp), allocatable :: threshold
  contains
    procedure :: fit=>fixed_fit
    procedure :: predict=>fixed_predict
  end type
  type, extends(threshold_classifier_base) :: tuned_threshold_classifier_cv
    integer :: cv=5,n_thresholds=100,random_state=0,positive_label_=0
    real(dp) :: validation_fraction=-1,best_threshold_=0,best_score_=0
    logical :: refit=.true.,store_cv_results=.false.
    character(20) :: scoring='balanced_accuracy',response_method_used_=''
    real(dp), allocatable :: thresholds(:),thresholds_(:),mean_test_score_(:),fold_thresholds_(:,:),fold_scores_(:,:)
    type(index_split), allocatable :: splits(:),cv_splits_(:)
  contains
    procedure :: fit=>tuned_fit
    procedure :: predict=>tuned_predict
  end type
  abstract interface
    function threshold_prediction(self,x) result(y)
      import threshold_classifier_base,dp
      class(threshold_classifier_base), intent(in) :: self
      real(dp), intent(in) :: x(:,:)
      integer, allocatable :: y(:)
    end function
    real(dp) function threshold_objective(y,predicted,sample_weight) result(value)
      import dp
      integer, intent(in) :: y(:),predicted(:)
      real(dp), intent(in), optional :: sample_weight(:)
    end function
  end interface
contains
  subroutine check_base(base)
    class(*), intent(in) :: base
    call require(base_supported(base,.true.),'threshold unsupported classifier base')
    call require(base_proba(base).or.base_decision(base),'threshold base has no continuous response')
  end subroutine
  subroutine check_binary(base,classes)
    class(*), intent(in) :: base
    integer, allocatable, intent(out) :: classes(:)
    call require(base_fitted(base),'threshold base is not fitted')
    classes=base_classes(base)
    call require(size(classes)==2,'threshold requires a binary fitted base')
  end subroutine
  subroutine response(base,x,method,positive,value,used)
    class(*), intent(in) :: base
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    integer, intent(in) :: positive
    real(dp), allocatable, intent(out) :: value(:)
    character(*), intent(out) :: used
    real(dp), allocatable :: matrix(:,:)
    integer, allocatable :: classes(:)
    integer :: column
    call check_binary(base,classes)
    call require(any(classes==positive),'threshold positive label is not a fitted class')
    used=method
    if(method=='auto') then
      used='decision_function'; if(base_proba(base)) used='predict_proba'
    end if
    select case(used)
    case('predict_proba')
      call require(base_proba(base),'threshold base has no predict_proba')
      matrix=probabilities(base,x); column=2; if(positive==classes(1)) column=1
      call require(size(matrix,2)==2,'threshold requires two probability columns')
      value=matrix(:,column)
    case('decision_function')
      call require(base_decision(base),'threshold base has no decision_function')
      matrix=decisions(base,x)
      call require(size(matrix,2)==1,'threshold requires a single binary decision column')
      value=matrix(:,1); if(positive==classes(1)) value=-value
    case default
      call require(.false.,'unknown threshold response method')
    end select
    call require(all(ieee_is_finite(value)),'threshold responses must be finite')
    call require(maxval(abs(value))<=huge(1._dp)/4,'threshold responses exceed working range')
  end subroutine
  function class_labels(values,cutoff,classes,positive) result(y)
    real(dp), intent(in) :: values(:),cutoff
    integer, intent(in) :: classes(:),positive
    integer :: y(size(values)),negative
    negative=classes(1); if(positive==classes(1)) negative=classes(2)
    y=merge(positive,negative,values>=cutoff)
  end function
  subroutine fit_base(destination,base,x,y,w,prefit)
    class(*), allocatable, intent(inout) :: destination
    class(*), intent(in) :: base
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: w(:)
    logical, intent(in) :: prefit
    call check_base(base); call check_weights(w,size(y))
    if(allocated(destination)) deallocate(destination)
    allocate(destination,source=base)
    if(prefit) then
      call require(base_fitted(base),'threshold prefit base is not fitted')
    else
      if(present(w)) call require(base_weighted(base),'threshold base has no weighted fit')
      call train_classifier(destination,x,y,w)
    end if
  end subroutine
  subroutine check_weights(w,n)
    real(dp), intent(in), optional :: w(:)
    integer, intent(in) :: n
    call score_weights(w,n,'threshold')
    if(present(w)) call require(all(w>=0),'threshold weights must be nonnegative')
  end subroutine
  subroutine fixed_fit(self,x,y,base,sample_weight)
    class(fixed_threshold_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: base
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: values(:)
    integer, allocatable :: classes(:)
    integer :: positive
    character(20) :: used
    call check_xy(x,size(y)); classes=unique_labels(y)
    call require(size(classes)==2,'threshold fit requires two target classes')
    call check_weights(sample_weight,size(y))
    if(allocated(self%threshold)) call require(.not.ieee_is_nan(self%threshold),'threshold cutoff must not be NaN')
    call fit_base(self%estimator_,base,x,y,sample_weight,self%prefit)
    call check_binary(self%estimator_,self%classes_); self%n_features_in_=size(x,2)
    positive=self%classes_(2); if(allocated(self%pos_label)) positive=self%pos_label
    call response(self%estimator_,x(:1,:),self%response_method,positive,values,used)
  end subroutine
  function fixed_predict(self,x) result(y)
    class(fixed_threshold_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: values(:)
    real(dp) :: cutoff
    integer :: positive
    character(20) :: used
    call check_model(self,x)
    positive=self%classes_(2); if(allocated(self%pos_label)) positive=self%pos_label
    call response(self%estimator_,x,self%response_method,positive,values,used)
    cutoff=0; if(used=='predict_proba') cutoff=.5_dp
    if(allocated(self%threshold)) then
      call require(.not.ieee_is_nan(self%threshold),'threshold cutoff must not be NaN'); cutoff=self%threshold
    end if
    y=class_labels(values,cutoff,self%classes_,positive)
  end function
  subroutine check_model(self,x)
    class(threshold_classifier_base), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%estimator_),'threshold classifier is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'threshold query width mismatch')
  end subroutine
  function threshold_proba(self,x) result(values)
    class(threshold_classifier_base), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    call check_model(self,x); call require(base_proba(self%estimator_),'threshold base has no predict_proba')
    values=probabilities(self%estimator_,x)
  end function
  function threshold_decision(self,x) result(values)
    class(threshold_classifier_base), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    call check_model(self,x); call require(base_decision(self%estimator_),'threshold base has no decision_function')
    values=decisions(self%estimator_,x)
  end function
  real(dp) function threshold_score(self,x,y,sample_weight) result(value)
    class(threshold_classifier_base), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call check_xy(x,size(y)); call score_weights(sample_weight,size(y),'threshold')
    value=ieee_mean(merge(1._dp,0._dp,y==self%predict(x)),sample_weight)
  end function
  subroutine make_splits(self,y)
    class(tuned_threshold_classifier_cv), intent(inout) :: self
    integer, intent(in) :: y(:)
    type(stratified_shuffle_split) :: shuffle
    integer, allocatable :: labels(:),counts(:),members(:),assignment(:)
    integer :: n,i,j,c,offset,position,width
    n=size(y)
    if(allocated(self%cv_splits_)) deallocate(self%cv_splits_)
    if(self%prefit) then
      allocate(self%cv_splits_(1)); self%cv_splits_(1)%test=[(i,i=1,n)]
      self%cv_splits_(1)%train=[integer::]
    else if(allocated(self%splits)) then
      self%cv_splits_=self%splits
      call require(size(self%cv_splits_)>0,'threshold CV needs at least one split')
    else if(self%validation_fraction/=-1) then
      call require(ieee_is_finite(self%validation_fraction),'threshold validation fraction must be finite')
      call require(self%validation_fraction>0.and.self%validation_fraction<1,'invalid threshold validation fraction')
      shuffle%n_splits=1; shuffle%test_size=self%validation_fraction; shuffle%random_state=self%random_state
      self%cv_splits_=shuffle%split(y)
    else
      call require(self%cv>=2.and.self%cv<=n,'invalid threshold CV count')
      labels=[integer::]
      do i=1,n
        if(.not.any(labels==y(i))) labels=[labels,y(i)]
      end do
      counts=[(count(y==labels(i)),i=1,size(labels))]
      call require(.not.all(counts<self%cv),'threshold CV exceeds every class count')
      allocate(self%cv_splits_(self%cv),assignment(n)); offset=0
      do c=1,size(labels)
        members=pack([(i,i=1,n)],y==labels(c)); position=1
        do j=1,self%cv
          width=count(modulo(offset+[(i,i=0,counts(c)-1)],self%cv)==j-1)
          assignment(members(position:position+width-1))=j; position=position+width
        end do
        offset=offset+counts(c)
      end do
      do j=1,self%cv
        self%cv_splits_(j)%train=pack([(i,i=1,n)],assignment/=j)
        self%cv_splits_(j)%test=pack([(i,i=1,n)],assignment==j)
      end do
    end if
    do i=1,size(self%cv_splits_)
      call require(allocated(self%cv_splits_(i)%train).and.allocated(self%cv_splits_(i)%test), &
        'threshold split indices are unallocated')
      call require(size(self%cv_splits_(i)%test)>0,'threshold split test indices must be nonempty')
      if(.not.self%prefit) call require(size(self%cv_splits_(i)%train)>0,'threshold split train indices must be nonempty')
      call require(all(self%cv_splits_(i)%train>=1).and.all(self%cv_splits_(i)%train<=n).and. &
        all(self%cv_splits_(i)%test>=1).and.all(self%cv_splits_(i)%test<=n),'threshold split index out of bounds')
    end do
  end subroutine
  function linear_grid(first,last,n) result(values)
    real(dp), intent(in) :: first,last
    integer, intent(in) :: n
    real(dp) :: values(n),step
    integer :: j
    values=first
    if(n>1) then
      step=(last-first)/real(n-1,dp)
      values=[(first+real(j-1,dp)*step,j=1,n)]; values(n)=last
    end if
  end function
  real(dp) function interpolate(x,xp,fp) result(value)
    real(dp), intent(in) :: x,xp(:),fp(:)
    integer :: left,right,middle
    if(x<xp(1)) then
      value=fp(1); return
    else if(x>=xp(size(xp))) then
      value=fp(size(fp)); return
    end if
    left=1; right=size(xp)
    do while(right-left>1)
      middle=(left+right)/2
      if(x>=xp(middle)) then
        left=middle
      else
        right=middle
      end if
    end do
    value=fp(left)
    if(x/=xp(left)) value=fp(left)+(fp(right)-fp(left))/(xp(right)-xp(left))*(x-xp(left))
  end function
  real(dp) function objective_value(y,predicted,positive,scoring,w) result(value)
    integer, intent(in) :: y(:),predicted(:),positive
    character(*), intent(in) :: scoring
    real(dp), intent(in), optional :: w(:)
    real(dp) :: weights(size(y)),tp,fp,fn,mass,recalls
    integer, allocatable :: classes(:)
    integer :: i,nused
    weights=1
    if(present(w)) weights=w/maxval(w)
    tp=sum(pack(weights,y==positive.and.predicted==positive))
    fp=sum(pack(weights,y/=positive.and.predicted==positive))
    fn=sum(pack(weights,y==positive.and.predicted/=positive))
    select case(scoring)
    case('accuracy')
      value=sum(pack(weights,y==predicted))/sum(weights)
    case('balanced_accuracy')
      classes=unique_labels(y); recalls=0; nused=0
      do i=1,size(classes)
        mass=sum(pack(weights,y==classes(i))); if(mass==0) cycle
        recalls=recalls+sum(pack(weights,y==classes(i).and.predicted==y))/mass; nused=nused+1
      end do
      value=recalls/nused
    case('precision')
      value=0; if(tp+fp>0) value=tp/(tp+fp)
    case('recall')
      value=0; if(tp+fn>0) value=tp/(tp+fn)
    case('f1')
      value=0; if(2*tp+fp+fn>0) value=2*tp/(2*tp+fp+fn)
    case('jaccard')
      value=0; if(tp+fp+fn>0) value=tp/(tp+fp+fn)
    case default
      call require(.false.,'unknown threshold scoring mode')
    end select
  end function
  subroutine tuned_fit(self,x,y,base,sample_weight,objective,score_weight)
    class(tuned_threshold_classifier_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: base
    real(dp), intent(in), optional :: sample_weight(:)
    procedure(threshold_objective), optional :: objective
    real(dp), intent(in), optional :: score_weight(:)
    class(*), allocatable :: worker
    integer, allocatable :: classes(:),train(:),test(:),predicted(:),worker_classes(:)
    real(dp), allocatable :: values(:),grid(:),scores(:,:),grids(:,:),mean_scores(:),weights(:),validation_weights(:)
    real(dp) :: lower,upper
    character(20) :: used
    integer :: n,f,j,nfolds,nt,best
    logical :: scoring_weighted
    call check_xy(x,size(y)); n=size(y); classes=unique_labels(y)
    call require(size(classes)==2,'threshold fit requires two target classes')
    call check_base(base); call check_weights(sample_weight,n); call check_weights(score_weight,n)
    scoring_weighted=present(sample_weight).or.present(score_weight)
    if(.not.present(objective)) call require(any(self%scoring==[character(20):: &
      'accuracy','balanced_accuracy','precision','recall','f1','jaccard']),'unknown threshold scoring mode')
    if(allocated(self%thresholds)) then
      nt=size(self%thresholds)
      call require(nt>0,'threshold grid must be nonempty')
      call require(all(ieee_is_finite(self%thresholds)),'threshold grid must be finite')
      call require(maxval(abs(self%thresholds))<=huge(1._dp)/4,'threshold grid exceeds working range')
      call require(all(self%thresholds(2:)>=self%thresholds(:nt-1)),'threshold grid must be nondecreasing')
    else
      nt=self%n_thresholds; call require(nt>0,'threshold grid count must be positive')
    end if
    call require(.not.(self%prefit.and.self%refit),'threshold prefit requires refit=false')
    call make_splits(self,y); nfolds=size(self%cv_splits_)
    call require(self%refit.or.nfolds==1,'threshold refit=false requires a single split')
    allocate(weights(n),validation_weights(n)); weights=1; if(present(sample_weight)) weights=sample_weight
    validation_weights=weights; if(present(score_weight)) validation_weights=score_weight
    if(self%refit.or.self%prefit) then
      call fit_base(self%estimator_,base,x,y,sample_weight,self%prefit)
    else
      train=self%cv_splits_(1)%train
      if(present(sample_weight)) then
        call fit_base(self%estimator_,base,x(train,:),y(train),weights(train),.false.)
      else
        call fit_base(self%estimator_,base,x(train,:),y(train),prefit=.false.)
      end if
    end if
    call check_binary(self%estimator_,self%classes_); self%n_features_in_=size(x,2)
    call require(all(self%classes_==classes),'threshold target classes do not match fitted base')
    self%positive_label_=classes(2)
    if(.not.present(objective).and.any(self%scoring==[character(20)::'precision','recall','f1','jaccard'])) &
      self%positive_label_=1
    if(allocated(self%pos_label)) self%positive_label_=self%pos_label
    call require(any(classes==self%positive_label_),'threshold positive label is not a fitted class')
    allocate(grids(nfolds,nt),scores(nfolds,nt),mean_scores(nt))
    do f=1,nfolds
      train=self%cv_splits_(f)%train; test=self%cv_splits_(f)%test
      call require(sum(validation_weights(test)/maxval(validation_weights))>0,'threshold test weights require positive total')
      if(self%prefit) then
        allocate(worker,source=self%estimator_)
      else if(present(sample_weight)) then
        call fit_base(worker,base,x(train,:),y(train),weights(train),.false.)
      else
        call fit_base(worker,base,x(train,:),y(train),prefit=.false.)
      end if
      call check_binary(worker,worker_classes)
      call require(all(worker_classes==classes),'threshold fold classes do not match fitted base')
      call response(worker,x(test,:),self%response_method,self%positive_label_,values,used)
      self%response_method_used_=used
      if(allocated(self%thresholds)) then
        grid=self%thresholds
      else
        grid=linear_grid(minval(values),maxval(values),nt)
      end if
      call require(abs(grid(1)-grid(nt))>1.e-8_dp+1.e-5_dp*abs(grid(nt)), &
        'threshold curve endpoints are too close for optimization')
      grids(f,:)=grid
      do j=1,nt
        predicted=class_labels(values,grid(j),classes,self%positive_label_)
        if(present(objective)) then
          if(scoring_weighted) then
            scores(f,j)=objective(y(test),predicted,validation_weights(test))
          else
            scores(f,j)=objective(y(test),predicted)
          end if
        else if(scoring_weighted) then
          scores(f,j)=objective_value(y(test),predicted,self%positive_label_,self%scoring,validation_weights(test))
        else
          scores(f,j)=objective_value(y(test),predicted,self%positive_label_,self%scoring)
        end if
      end do
      call require(all(ieee_is_finite(scores(f,:))),'threshold objective scores must be finite')
      deallocate(worker)
    end do
    if(allocated(self%thresholds)) then
      grid=self%thresholds
    else
      lower=minval(grids); upper=maxval(grids); grid=linear_grid(lower,upper,nt)
    end if
    mean_scores=0
    do j=1,nt
      do f=1,nfolds
        mean_scores(j)=mean_scores(j)+interpolate(grid(j),grids(f,:),scores(f,:))
      end do
      mean_scores(j)=mean_scores(j)/nfolds
    end do
    best=maxloc(mean_scores,dim=1); self%best_threshold_=grid(best); self%best_score_=mean_scores(best)
    if(allocated(self%thresholds_)) deallocate(self%thresholds_,self%mean_test_score_,self%fold_thresholds_,self%fold_scores_)
    if(self%store_cv_results) then
      self%thresholds_=grid; self%mean_test_score_=mean_scores; self%fold_thresholds_=grids; self%fold_scores_=scores
    end if
  end subroutine
  function tuned_predict(self,x) result(y)
    class(tuned_threshold_classifier_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: values(:)
    character(20) :: used
    call check_model(self,x)
    call response(self%estimator_,x,self%response_method,self%positive_label_,values,used)
    y=class_labels(values,self%best_threshold_,self%classes_,self%positive_label_)
  end function
end module
