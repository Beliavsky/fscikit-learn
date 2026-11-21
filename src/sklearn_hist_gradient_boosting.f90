! Numeric histogram gradient boosting, sklearn 1.6.1 reference algorithms.
! Retained BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md.
module sklearn_hist_gradient_boosting
  use sklearn_common, only: dp,require,label_index,sigmoid,logsumexp
  use sklearn_metrics, only: r2_score,accuracy_score
  use sklearn_histogram_binning
  use sklearn_histogram_tree
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_total
  use, intrinsic :: iso_fortran_env, only: real32,int16,int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: hist_gradient_boosting_regressor,hist_gradient_boosting_classifier
  type, abstract :: histogram_boosting
    integer :: max_iter=100,max_leaf_nodes=31,max_depth=0,min_samples_leaf=20,max_bins=255,random_state=0
    integer :: n_iter_no_change=10,n_iter_=0,n_features_in_=0,n_trees_per_iteration_=0,n_classes_=0
    real(dp) :: learning_rate=.1_dp,l2_regularization=0,tol=1.e-7_dp
    logical :: warm_start=.false.,early_stopping=.false.,early_stopped_=.false.
    integer, allocatable :: monotonic_cst(:)
    type(histogram_bin_mapper) :: bin_mapper_
    type(histogram_tree), allocatable :: predictors_(:,:)
    real(dp), allocatable :: baseline_prediction_(:),train_score_(:)
    character(24), private :: loss_=''
    real(dp), private :: quantile_=.5_dp
    integer(int64), private :: bin_seed_=0
  contains
    procedure :: raw_predict=>boost_raw
    procedure :: staged_raw_predict=>boost_staged_raw
  end type
  type, extends(histogram_boosting) :: hist_gradient_boosting_regressor
    character(24) :: loss='squared_error'
    real(dp) :: quantile=.5_dp
  contains
    procedure :: fit=>regressor_fit
    procedure :: predict=>regressor_predict
    procedure :: staged_predict=>regressor_staged
    procedure :: score=>regressor_score
  end type
  type, extends(histogram_boosting) :: hist_gradient_boosting_classifier
    character(24) :: loss='log_loss',class_weight='none'
    real(dp), allocatable :: class_weights(:)
    integer, allocatable :: classes_(:)
  contains
    procedure :: fit=>classifier_fit
    procedure :: predict=>classifier_predict
    procedure :: predict_proba=>classifier_proba
    procedure :: decision_function=>boost_decision
    procedure :: staged_predict=>classifier_staged
    procedure :: staged_predict_proba=>classifier_staged_proba
    procedure :: score=>classifier_score
  end type
contains
  subroutine check_features(x,width)
    real(dp), intent(in) :: x(:,:)
    integer, optional, intent(in) :: width
    call require(size(x,1)>0.and.size(x,2)>0,'HistGradientBoosting X must be nonempty')
    if(present(width)) call require(size(x,2)==width,'HistGradientBoosting feature width mismatch')
    ! Numeric histograms accept NaN and +/-infinity, as in sklearn.
  end subroutine
  subroutine controls(self,p)
    class(histogram_boosting), intent(in) :: self
    integer, intent(in) :: p
    call require(self%max_iter>0,'HistGradientBoosting max_iter must be positive')
    call require(self%max_leaf_nodes==0.or.self%max_leaf_nodes>=2,'HistGradientBoosting invalid max_leaf_nodes')
    call require(self%max_depth>=0,'HistGradientBoosting max_depth must be nonnegative')
    call require(self%min_samples_leaf>0,'HistGradientBoosting invalid min_samples_leaf')
    call require(self%max_bins>=2.and.self%max_bins<=255,'HistGradientBoosting max_bins must be in [2,255]')
    call require(ieee_is_finite(self%learning_rate),'HistGradientBoosting learning_rate must be finite')
    call require(self%learning_rate>0,'HistGradientBoosting learning_rate must be positive')
    call require(ieee_is_finite(self%l2_regularization),'HistGradientBoosting regularization must be finite')
    call require(self%l2_regularization>=0,'HistGradientBoosting regularization must be nonnegative')
    call require(ieee_is_finite(self%tol),'HistGradientBoosting tol must be finite')
    call require(self%tol>=0,'HistGradientBoosting tol must be nonnegative')
    call require(self%n_iter_no_change>0,'HistGradientBoosting n_iter_no_change must be positive')
    call require(self%random_state>=0,'HistGradientBoosting random_state must be nonnegative')
    if(allocated(self%monotonic_cst)) then
      call require(size(self%monotonic_cst)==p,'HistGradientBoosting monotonic constraint width mismatch')
      call require(all(self%monotonic_cst>=-1).and.all(self%monotonic_cst<=1), &
        'HistGradientBoosting monotonic constraints must be -1,0,1')
    end if
  end subroutine
  function weights(n,sample_weight) result(w)
    integer, intent(in) :: n
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: w(n)
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'HistGradientBoosting weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'HistGradientBoosting weights must be finite')
      call require(all(sample_weight>=0),'HistGradientBoosting weights must be nonnegative')
      w=sample_weight
    end if
    call require(ieee_is_finite(ieee_total(w)),'HistGradientBoosting weight sum overflow')
    call require(sum(w)>0,'HistGradientBoosting requires positive total weight')
  end function
  real(dp) function weighted_mean(y,w) result(mean)
    real(dp), intent(in) :: y(:),w(:)
    real(dp) :: products(size(y))
    integer :: i
    do i=1,size(y)
      products(i)=ieee_product(y(i),w(i))
    end do
    mean=ieee_total(products)/sum(w)
    call require(ieee_is_finite(mean),'HistGradientBoosting baseline exceeds finite working range')
  end function
  real(dp) function percentile(y,w,q,weighted,median) result(value)
    real(dp), intent(in) :: y(:),w(:),q
    logical, intent(in) :: weighted,median
    integer, allocatable :: order(:)
    real(dp), allocatable :: cumulative(:)
    real(dp) :: position,fraction,difference,target
    integer :: j,a,b,n
    n=size(y);order=histogram_argsort(y)
    if(weighted) then
      allocate(cumulative(n));cumulative(1)=w(order(1))
      do j=2,n
        cumulative(j)=cumulative(j-1)+w(order(j))
      end do
      target=q*cumulative(n);j=1
      do while(j<n)
        if(target==0) then
          if(cumulative(j)>0) exit
        else
          if(cumulative(j)>=target) exit
        end if
        j=j+1
      end do
      value=y(order(j))
    else if(median) then
      a=(n+1)/2;b=n/2+1;value=y(order(a))
      if(a/=b) value=.5_dp*ieee_difference(y(order(a)),-y(order(b)))
    else
      position=(n-1)*((100*q)/100);a=floor(position)+1;b=min(n,a+1);fraction=position-floor(position)
      difference=ieee_difference(y(order(b)),y(order(a)))
      if(fraction>=.5_dp) then
        value=ieee_difference(y(order(b)),difference*(1-fraction))
      else
        value=y(order(a))+difference*fraction
      end if
    end if
  end function
  function binned_apply(tree,bins,missing) result(ids)
    type(histogram_tree), intent(in) :: tree
    integer(int16), intent(in) :: bins(:,:)
    integer, intent(in) :: missing
    integer :: ids(size(bins,1)),i,node,f,b
    logical :: left
    do i=1,size(bins,1)
      node=1
      do while(tree%nodes_(node)%left/=0)
        f=tree%nodes_(node)%feature;b=int(bins(i,f));left=b<=tree%nodes_(node)%bin_threshold
        if(b==missing) left=tree%nodes_(node)%missing_go_to_left
        if(left) then
          node=tree%nodes_(node)%left
        else
          node=tree%nodes_(node)%right
        end if
      end do
      ids(i)=node
    end do
  end function
  subroutine gradients(loss,quantile,y,raw,w,g,h)
    character(*), intent(in) :: loss
    real(dp), intent(in) :: quantile,y(:),raw(:,:),w(:)
    real(real32), intent(out) :: g(:,:),h(:,:)
    real(dp) :: gd(size(raw,1),size(raw,2)),hd(size(raw,1),size(raw,2)),prob(size(raw,2)),t,mu
    integer :: i,k,label
    do i=1,size(y)
      select case(loss)
      case('squared_error');gd(i,1)=ieee_difference(raw(i,1),y(i));hd(i,1)=1
      case('absolute_error');gd(i,1)=merge(1._dp,-1._dp,raw(i,1)>y(i));hd(i,1)=1
      case('quantile');gd(i,1)=merge(1-quantile,-quantile,raw(i,1)>y(i));hd(i,1)=1
      case('poisson')
        call require(raw(i,1)<log(huge(1._dp)),'HistGradientBoosting exponential prediction overflow')
        mu=exp(raw(i,1));gd(i,1)=mu-y(i);hd(i,1)=mu
      case('gamma')
        call require(-raw(i,1)<log(huge(1._dp)),'HistGradientBoosting exponential prediction overflow')
        t=ieee_product(y(i),exp(-raw(i,1)));gd(i,1)=1-t;hd(i,1)=t
      case('binary')
        if(raw(i,1)>=0) then
          t=exp(-raw(i,1));gd(i,1)=((1-y(i))-y(i)*t)/(1+t)
        else
          t=exp(raw(i,1));gd(i,1)=((1-y(i))*t-y(i))/(1+t)
        end if
        hd(i,1)=t/(1+t)**2
      case('multiclass')
        prob=exp(raw(i,:)-maxval(raw(i,:)));prob=prob/sum(prob);label=int(y(i))+1
        gd(i,:)=prob;gd(i,label)=gd(i,label)-1;hd(i,:)=prob*(1-prob)
      end select
      if(w(i)/=1) then
        do k=1,size(raw,2)
          gd(i,k)=ieee_product(gd(i,k),w(i));hd(i,k)=ieee_product(hd(i,k),w(i))
        end do
      end if
    end do
    call require(all(ieee_is_finite(gd)).and.all(ieee_is_finite(hd)), &
      'HistGradientBoosting nonfinite gradient or Hessian')
    call require(maxval(abs(gd))<=real(huge(1._real32),dp).and.maxval(abs(hd))<=real(huge(1._real32),dp), &
      'HistGradientBoosting gradients exceed float32 range')
    g=real(gd,real32);h=real(hd,real32)
  end subroutine
  real(dp) function source_softplus(x) result(value)
    real(dp), intent(in) :: x
    real(dp) :: t
    ! Same cutoffs as sklearn _loss.pyx log1pexp. The atanh identity computes
    ! log1p(t) without cancellation and uses only standard Fortran intrinsics.
    if(x<=-37) then
      value=exp(x)
    else if(x<=-2) then
      t=exp(x);value=2*atanh(t/(2+t))
    else if(x<=18) then
      value=log(1+exp(x))
    else if(x<=33.3_dp) then
      value=x+exp(-x)
    else
      value=x
    end if
  end function
  real(dp) function loss_score(loss,quantile,y,raw,w) result(score)
    character(*), intent(in) :: loss
    real(dp), intent(in) :: quantile,y(:),raw(:,:),w(:)
    real(dp) :: values(size(y)),r
    integer :: i
    do i=1,size(y)
      r=raw(i,1)
      select case(loss)
      case('squared_error');values(i)=.5_dp*(r-y(i))**2
      case('absolute_error');values(i)=abs(r-y(i))
      case('quantile');values(i)=max(quantile*(y(i)-r),(1-quantile)*(r-y(i)))
      case('poisson');values(i)=exp(r)-y(i)*r
      case('gamma');values(i)=r+y(i)*exp(-r)
      case('binary');values(i)=source_softplus(r)-y(i)*r
      case('multiclass');values(i)=logsumexp(raw(i,:))-raw(i,int(y(i))+1)
      end select
    end do
    score=-sum(w*values)/sum(w)
  end function
  subroutine fit_core(self,x,y,w,weighted,loss,quantile,nclasses)
    class(histogram_boosting), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),w(:),quantile
    logical, intent(in) :: weighted
    character(*), intent(in) :: loss
    integer, intent(in) :: nclasses
    type(histogram_tree), allocatable :: models(:,:)
    type(legacy_random_state) :: rng
    integer(int16), allocatable :: bins(:,:)
    real(real32), allocatable :: g(:,:),h(:,:)
    real(dp), allocatable :: raw(:,:),scores(:),prior(:),residual(:)
    integer, allocatable :: monotonic(:),ids(:),rows(:)
    integer :: n,p,nt,old,stage,k,j,nhist,reference
    real(dp) :: mean,eps
    logical :: warm
    call check_features(x);n=size(x,1);p=size(x,2);call controls(self,p)
    call require(size(y)==n.and.size(w)==n,'HistGradientBoosting target count mismatch')
    call require(all(ieee_is_finite(y)),'HistGradientBoosting y must be finite')
    nt=1;if(nclasses>2) nt=nclasses
    call require(int(self%max_iter,int64)*nt<huge(1),'HistGradientBoosting predictor count exceeds integer capacity')
    allocate(monotonic(p));monotonic=0;if(allocated(self%monotonic_cst)) monotonic=self%monotonic_cst
    if(nclasses>2) call require(.not.allocated(self%monotonic_cst), &
      'HistGradientBoosting multiclass monotonicity is unsupported')
    warm=self%warm_start.and.allocated(self%predictors_);old=0;nhist=0
    if(warm) then
      call require(self%n_features_in_==p,'HistGradientBoosting warm-start width changed')
      call require(self%loss_==loss.and.self%n_classes_==nclasses,'HistGradientBoosting warm-start loss/classes changed')
      call require(self%max_iter>=self%n_iter_,'HistGradientBoosting warm-start iteration count decreased')
      old=self%n_iter_
    else
      call rng%seed(self%random_state)
      do
        self%bin_seed_=rng%raw_uint32();if(self%bin_seed_<4294967295_int64) exit
      end do
      self%n_features_in_=p;self%n_trees_per_iteration_=nt;self%n_classes_=nclasses
      self%loss_=loss;self%quantile_=quantile;self%n_iter_=0
      if(allocated(self%baseline_prediction_)) deallocate(self%baseline_prediction_)
      allocate(self%baseline_prediction_(nt));eps=10*epsilon(1._dp)
      if(loss=='multiclass') then
        allocate(prior(nt))
        do k=1,nt
          prior(k)=max(eps,min(1-eps,sum(pack(w,int(y)==k-1))/sum(w)))
        end do
        prior=log(prior);self%baseline_prediction_=prior-sum(prior)/nt
      else
        select case(loss)
        case('absolute_error');mean=percentile(y,w,.5_dp,weighted,.true.)
        case('quantile');mean=percentile(y,w,quantile,weighted,.false.)
        case default
          mean=weighted_mean(y,w)
          if(loss=='binary') then
            mean=max(eps,min(1-eps,mean));mean=log(mean/(1-mean))
          else if(loss=='poisson'.or.loss=='gamma') then
            mean=log(max(eps,mean))
          end if
        end select
        call require(ieee_is_finite(mean),'HistGradientBoosting baseline exceeds finite working range')
        self%baseline_prediction_=mean
      end if
      if(allocated(self%train_score_)) deallocate(self%train_score_)
      allocate(self%train_score_(0))
    end if
    self%bin_mapper_%max_bins=self%max_bins;call self%bin_mapper_%fit(x,self%bin_seed_)
    bins=self%bin_mapper_%transform(x)
    allocate(models(self%max_iter,nt),raw(n,nt),g(n,nt),h(n,nt),scores(self%max_iter+1))
    raw=spread(self%baseline_prediction_,1,n)
    if(warm) then
      models(:old,:)=self%predictors_
      do stage=1,old
        do k=1,nt
          ids=binned_apply(models(stage,k),bins,self%bin_mapper_%missing_bin_)
          do j=1,n
            raw(j,k)=raw(j,k)+models(stage,k)%nodes_(ids(j))%value
          end do
        end do
      end do
      nhist=size(self%train_score_)
      if(nhist>0) scores(:nhist)=self%train_score_
    else if(self%early_stopping) then
      nhist=1;scores(1)=loss_score(loss,quantile,y,raw,w)
    end if
    self%early_stopped_=.false.
    do stage=old+1,self%max_iter
      call gradients(loss,quantile,y,raw,w,g,h)
      do k=1,nt
        models(stage,k)=grow_histogram_tree(bins,g(:,k),h(:,k),self%bin_mapper_,self%min_samples_leaf, &
          self%max_leaf_nodes,self%max_depth,self%l2_regularization,self%learning_rate,monotonic)
        ids=binned_apply(models(stage,k),bins,self%bin_mapper_%missing_bin_)
        if(loss=='absolute_error'.or.loss=='quantile') then
          residual=y-raw(:,k)
          do j=1,size(models(stage,k)%nodes_)
            if(models(stage,k)%nodes_(j)%left/=0) cycle
            rows=pack([(reference,reference=1,n)],ids==j)
            mean=.5_dp;if(loss=='quantile') mean=quantile
            models(stage,k)%nodes_(j)%value=self%learning_rate* &
              percentile(residual(rows),w(rows),mean,weighted,loss=='absolute_error')
          end do
        end if
        do j=1,n
          raw(j,k)=raw(j,k)+models(stage,k)%nodes_(ids(j))%value
        end do
      end do
      self%n_iter_=stage
      if(self%early_stopping) then
        nhist=nhist+1;scores(nhist)=loss_score(loss,quantile,y,raw,w)
        if(nhist>self%n_iter_no_change) then
          reference=nhist-self%n_iter_no_change
          if(.not.any(scores(reference+1:nhist)>scores(reference)+self%tol)) then
            self%early_stopped_=.true.;exit
          end if
        end if
      end if
    end do
    self%predictors_=models(:self%n_iter_,:);self%train_score_=scores(:nhist)
  end subroutine
  subroutine regressor_fit(self,x,y,sample_weight)
    class(hist_gradient_boosting_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: w(:)
    call check_features(x)
    call require(any(self%loss==[character(24)::'squared_error','absolute_error','poisson','gamma','quantile']), &
      'HistGradientBoostingRegressor unknown loss')
    call require(ieee_is_finite(self%quantile),'HistGradientBoosting quantile must be finite')
    call require(self%quantile>=0.and.self%quantile<=1,'HistGradientBoosting quantile must be in [0,1]')
    call require(all(ieee_is_finite(y)),'HistGradientBoosting y must be finite')
    if(self%loss=='poisson') call require(all(y>=0).and.any(y>0),'HistGradientBoosting Poisson y domain')
    if(self%loss=='gamma') call require(all(y>0),'HistGradientBoosting Gamma y domain')
    w=weights(size(x,1),sample_weight)
    call fit_core(self,x,y,w,present(sample_weight),self%loss,self%quantile,0)
  end subroutine
  subroutine classifier_fit(self,x,y,sample_weight)
    class(hist_gradient_boosting_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: encoded(:),w(:),cw(:)
    integer, allocatable :: classes(:),order(:)
    integer :: i,k
    call check_features(x)
    call require(self%loss=='log_loss','HistGradientBoostingClassifier requires log_loss')
    call require(self%class_weight=='none'.or.self%class_weight=='balanced','invalid histogram class_weight')
    call require(size(y)==size(x,1),'HistGradientBoosting target count mismatch')
    order=histogram_argsort(real(y,dp));allocate(classes(size(y)));k=0
    do i=1,size(y)
      if(i>1) then
        if(y(order(i))==y(order(i-1))) cycle
      end if
      k=k+1;classes(k)=y(order(i))
    end do
    classes=classes(:k)
    if(self%warm_start.and.allocated(self%predictors_)) then
      call require(size(classes)==size(self%classes_),'HistGradientBoosting warm-start classes changed')
      call require(all(classes==self%classes_),'HistGradientBoosting warm-start classes changed')
    end if
    self%classes_=classes;allocate(encoded(size(y)),cw(size(classes)));cw=1
    w=weights(size(y),sample_weight)
    if(self%class_weight=='balanced') then
      do k=1,size(classes)
        cw(k)=real(size(y),dp)/(size(classes)*real(count(y==classes(k)),dp))
      end do
    end if
    if(allocated(self%class_weights)) then
      call require(size(self%class_weights)==size(classes),'histogram class weight width mismatch')
      call require(all(ieee_is_finite(self%class_weights)),'histogram class weights must be finite')
      call require(all(self%class_weights>=0),'histogram class weights must be nonnegative')
      cw=self%class_weights
    end if
    do i=1,size(y)
      k=label_index(classes,y(i));encoded(i)=k-1;w(i)=ieee_product(w(i),cw(k))
    end do
    call require(all(ieee_is_finite(w)).and.ieee_is_finite(ieee_total(w)),'histogram class weight overflow')
    call require(sum(w)>0,'histogram class weights require positive total weight')
    if(size(classes)<=2) then
      call fit_core(self,x,encoded,w,.true.,'binary',.5_dp,size(classes))
    else
      call fit_core(self,x,encoded,w,.true.,'multiclass',.5_dp,size(classes))
    end if
  end subroutine
  function boost_raw(self,x) result(raw)
    class(histogram_boosting), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: raw(:,:)
    integer :: stage,k
    call require(allocated(self%predictors_),'HistGradientBoosting is not fitted')
    call check_features(x,self%n_features_in_)
    raw=spread(self%baseline_prediction_,1,size(x,1))
    do stage=1,self%n_iter_
      do k=1,self%n_trees_per_iteration_
        raw(:,k)=raw(:,k)+self%predictors_(stage,k)%predict(x)
      end do
    end do
  end function
  function boost_staged_raw(self,x) result(raw)
    class(histogram_boosting), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: raw(:,:,:),current(:,:)
    integer :: stage,k
    call require(allocated(self%predictors_),'HistGradientBoosting is not fitted')
    call check_features(x,self%n_features_in_)
    allocate(raw(size(x,1),self%n_trees_per_iteration_,self%n_iter_))
    current=spread(self%baseline_prediction_,1,size(x,1))
    do stage=1,self%n_iter_
      do k=1,self%n_trees_per_iteration_
        current(:,k)=current(:,k)+self%predictors_(stage,k)%predict(x)
      end do
      raw(:,:,stage)=current
    end do
  end function
  function regressor_predict(self,x) result(prediction)
    class(hist_gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prediction(:),raw(:,:)
    raw=self%raw_predict(x);prediction=raw(:,1)
    if(self%loss_=='poisson'.or.self%loss_=='gamma') then
      call require(all(prediction<log(huge(1._dp))),'HistGradientBoosting prediction overflow')
      prediction=exp(prediction)
    end if
  end function
  function regressor_staged(self,x) result(prediction)
    class(hist_gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prediction(:,:),raw(:,:,:)
    raw=self%staged_raw_predict(x);prediction=raw(:,1,:)
    if(self%loss_=='poisson'.or.self%loss_=='gamma') then
      call require(all(prediction<log(huge(1._dp))),'HistGradientBoosting prediction overflow')
      prediction=exp(prediction)
    end if
  end function
  real(dp) function regressor_score(self,x,y) result(score)
    class(hist_gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    score=r2_score(y,self%predict(x))
  end function
  function probabilities(raw) result(proba)
    real(dp), intent(in) :: raw(:,:)
    real(dp), allocatable :: proba(:,:)
    integer :: i,k
    k=size(raw,2);allocate(proba(size(raw,1),max(2,k)))
    if(k==1) then
      proba(:,2)=sigmoid(raw(:,1));proba(:,1)=1-proba(:,2)
    else
      do i=1,size(raw,1)
        proba(i,:)=exp(raw(i,:)-maxval(raw(i,:)));proba(i,:)=proba(i,:)/sum(proba(i,:))
      end do
    end if
  end function
  function classifier_proba(self,x) result(proba)
    class(hist_gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: proba(:,:)
    proba=probabilities(self%raw_predict(x))
  end function
  function classifier_predict(self,x) result(prediction)
    class(hist_gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: prediction(:)
    real(dp), allocatable :: raw(:,:)
    integer :: i,k
    raw=self%raw_predict(x);allocate(prediction(size(x,1)))
    do i=1,size(prediction)
      k=maxloc(raw(i,:),dim=1)
      if(size(raw,2)==1) k=merge(2,1,raw(i,1)>0)
      prediction(i)=self%classes_(min(k,size(self%classes_)))
    end do
  end function
  function boost_decision(self,x) result(raw)
    class(hist_gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: raw(:,:)
    raw=self%raw_predict(x)
  end function
  function classifier_staged_proba(self,x) result(proba)
    class(hist_gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: proba(:,:,:),raw(:,:,:)
    integer :: stage
    raw=self%staged_raw_predict(x);allocate(proba(size(x,1),max(2,size(raw,2)),size(raw,3)))
    do stage=1,size(raw,3)
      proba(:,:,stage)=probabilities(raw(:,:,stage))
    end do
  end function
  function classifier_staged(self,x) result(prediction)
    class(hist_gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: prediction(:,:)
    real(dp), allocatable :: raw(:,:,:)
    integer :: i,j,k
    raw=self%staged_raw_predict(x);allocate(prediction(size(x,1),self%n_iter_))
    do j=1,self%n_iter_
      do i=1,size(x,1)
        k=maxloc(raw(i,:,j),dim=1)
        if(size(raw,2)==1) k=merge(2,1,raw(i,1,j)>0)
        prediction(i,j)=self%classes_(min(k,size(self%classes_)))
      end do
    end do
  end function
  real(dp) function classifier_score(self,x,y) result(score)
    class(hist_gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    score=accuracy_score(y,self%predict(x))
  end function
end module
