module sklearn_linear_model
  use sklearn_omp_cv, only: orthogonal_matching_pursuit_cv
  use sklearn_least_angle, only: lars,lasso_lars,lars_path,lars_path_result, &
    lars_cv,lasso_lars_cv,lasso_lars_ic
  use sklearn_sgd_one_class, only: sgd_one_class_svm
  use sklearn_quantile, only: quantile_regressor
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_multitask, only: multi_task_elastic_net,multi_task_lasso,multi_task_elastic_net_cv,multi_task_lasso_cv
  use sklearn_ridge_classification, only: ridge_classifier,ridge_classifier_cv
  use sklearn_common
  use sklearn_metrics, only: r2_score,accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: orthogonal_matching_pursuit_cv
  public :: lars,lasso_lars,lars_path,lars_path_result,lars_cv,lasso_lars_cv,lasso_lars_ic
  public :: sgd_one_class_svm
  public :: quantile_regressor
  public :: sgd_regressor,sgd_classifier
  public :: multi_task_elastic_net,multi_task_lasso,multi_task_elastic_net_cv,multi_task_lasso_cv
  public :: ridge_classifier,ridge_classifier_cv
  public :: elastic_net_cv,lasso_cv
  public :: linear_regression, ridge, lasso, elastic_net, logistic_regression, perceptron
  public :: bayesian_ridge, ard_regression, orthogonal_matching_pursuit
  public :: huber_regressor,ransac_regressor,tweedie_regressor,poisson_regressor,gamma_regressor
  public :: theil_sen_regressor
  public :: passive_aggressive_classifier,passive_aggressive_regressor
  type :: pa_config
    real(dp) :: c=1,tol=1.e-3_dp,epsilon=0.1_dp
    integer :: max_iter=1000,n_iter_no_change=5,average=0
    logical :: shuffle=.true.,fit_intercept=.true.,classification=.false.,pa_two=.false.
  end type
  type :: linear_regression
    logical :: fit_intercept=.true.
    real(dp), allocatable :: coef_(:)
    real(dp) :: intercept_=0
    integer :: rank_=0
  contains
    procedure :: fit=>ols_fit
    procedure :: predict=>linear_predict
    procedure :: score=>linear_score
  end type
  type, extends(linear_regression) :: ridge
    real(dp) :: alpha=1
  contains
    procedure :: fit=>ridge_fit
  end type
  type, extends(linear_regression) :: elastic_net
    real(dp) :: alpha=1, l1_ratio=0.5_dp,tol=1.e-8_dp
    integer :: max_iter=10000,n_iter_=0
    logical :: converged_=.false.
  contains
    procedure :: fit=>elastic_fit
  end type
  type, extends(elastic_net) :: lasso
  contains
    procedure :: fit=>lasso_fit
  end type
  ! CV contracts: sklearn 1.6.1 linear_model/_coordinate_descent.py (BSD).
  type, extends(elastic_net) :: elastic_net_cv
    integer :: cv=5,n_alphas=100,n_features_in_=0
    real(dp) :: eps=1.e-3_dp,alpha_=0,l1_ratio_=0,best_mse_=0
    real(dp), allocatable :: alphas(:),l1_ratios(:),alphas_(:,:),mse_path_(:,:,:)
    logical :: all_cv_converged_=.false.
  contains
    procedure :: fit=>elastic_cv_fit
  end type
  type, extends(elastic_net_cv) :: lasso_cv
  contains
    procedure :: fit=>lasso_cv_fit
  end type
  type :: logistic_regression
    logical :: fit_intercept=.true., converged_=.false.
    real(dp) :: c=1, tol=1.e-8_dp
    integer :: max_iter=200,n_iter_=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>logistic_fit
    procedure :: decision_function=>logistic_decision
    procedure :: predict_proba=>logistic_proba
    procedure :: predict=>logistic_predict
    procedure :: score=>logistic_score
  end type
  type :: perceptron
    logical :: fit_intercept=.true.
    integer :: max_iter=1000, n_iter_=0
    real(dp) :: eta0=1
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>perceptron_fit
    procedure :: decision_function=>perceptron_decision
    procedure :: predict=>perceptron_predict
    procedure :: score=>perceptron_score
  end type
  type :: passive_aggressive_classifier
    real(dp) :: c=1,tol=1.e-3_dp,t_=1
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0
    character(16) :: loss='hinge'
    logical :: fit_intercept=.true.,shuffle=.true.,class_weight_balanced=.false.,converged_=.false.
    integer, allocatable :: classes_(:),n_iter_per_class_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:),loss_(:)
  contains
    procedure :: fit=>pa_classifier_fit
    procedure :: decision_function=>pa_classifier_decision
    procedure :: predict=>pa_classifier_predict
    procedure :: score=>pa_classifier_score
  end type
  type, extends(linear_regression) :: passive_aggressive_regressor
    real(dp) :: c=1,tol=1.e-3_dp,epsilon=0.1_dp,t_=1,loss_=0
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0
    character(32) :: loss='epsilon_insensitive'
    logical :: shuffle=.true.,converged_=.false.
  contains
    procedure :: fit=>pa_regressor_fit
  end type
  type, extends(linear_regression) :: bayesian_ridge
    integer :: max_iter=300,n_iter_=0
    real(dp) :: tol=1.e-3_dp,alpha_1=1.e-6_dp,alpha_2=1.e-6_dp
    real(dp) :: lambda_1=1.e-6_dp,lambda_2=1.e-6_dp,alpha_init=-1,lambda_init=1
    real(dp) :: alpha_=0,lambda_=0
    logical :: converged_=.false.
    real(dp), allocatable :: sigma_(:,:)
  contains
    procedure :: fit=>bayesian_fit
    procedure :: predict_std=>bayesian_std
  end type
  type, extends(linear_regression) :: ard_regression
    integer :: max_iter=300,n_iter_=0
    real(dp) :: tol=1.e-3_dp,alpha_1=1.e-6_dp,alpha_2=1.e-6_dp
    real(dp) :: lambda_1=1.e-6_dp,lambda_2=1.e-6_dp,threshold_lambda=1.e4_dp,alpha_=0
    logical :: converged_=.false.
    real(dp), allocatable :: lambda_(:),sigma_(:,:)
  contains
    procedure :: fit=>ard_fit
    procedure :: predict_std=>ard_std
  end type
  type, extends(linear_regression) :: orthogonal_matching_pursuit
    integer :: n_nonzero_coefs=0,n_iter_=0
    real(dp) :: tol=-1
  contains
    procedure :: fit=>omp_fit
  end type
  type, extends(linear_regression) :: huber_regressor
    real(dp) :: epsilon=1.35_dp,alpha=1.e-4_dp,tol=1.e-5_dp,scale_=1,loss_=0
    integer :: max_iter=100,n_iter_=0
    logical :: converged_=.false.
    logical, allocatable :: outliers_(:)
  contains
    procedure :: fit=>huber_fit
  end type
  type, extends(linear_regression) :: ransac_regressor
    integer :: min_samples=0,max_trials=100,random_state=0,stop_n_inliers=huge(1)
    integer :: n_trials_=0,n_skips_no_inliers_=0
    real(dp) :: residual_threshold=-1,stop_probability=0.99_dp,stop_score=huge(1._dp)
    character(20) :: loss='absolute_error'
    logical, allocatable :: inlier_mask_(:)
  contains
    procedure :: fit=>ransac_fit
  end type
  type, extends(linear_regression) :: theil_sen_regressor
    integer :: n_subsamples=0,max_subpopulation=10000,max_iter=300,random_state=0
    integer :: n_subsamples_=0,n_subpopulation_=0,n_iter_=0
    real(dp) :: tol=1.e-3_dp,breakdown_=0
    logical :: exhaustive_=.false.,converged_=.false.
    integer, allocatable :: subsets_(:,:)
  contains
    procedure :: fit=>theil_sen_fit
  end type
  type, extends(linear_regression) :: tweedie_regressor
    real(dp) :: power=0,alpha=1,tol=1.e-4_dp
    integer :: max_iter=100,n_iter_=0
    character(12) :: link='auto',link_='identity'
    logical :: converged_=.false.
  contains
    procedure :: fit=>tweedie_fit
    procedure :: predict=>glm_predict
    procedure :: score=>glm_score
  end type
  type, extends(tweedie_regressor) :: poisson_regressor
  contains
    procedure :: fit=>poisson_fit
  end type
  type, extends(tweedie_regressor) :: gamma_regressor
  contains
    procedure :: fit=>gamma_fit
  end type
contains
  ! sklearn 1.6.1 linear_model/_passive_aggressive.py, _stochastic_gradient.py
  ! and _sgd_fast.pyx.tp. Retained BSD terms: LICENSES/scikit-learn-COPYING.
  function pa_settings(c,tol,max_iter,patience,average,shuffle,fit_intercept) result(options)
    real(dp), intent(in) :: c,tol
    integer, intent(in) :: max_iter,patience,average
    logical, intent(in) :: shuffle,fit_intercept
    type(pa_config) :: options
    call require(all(ieee_is_finite([c,tol])),'nonfinite passive-aggressive controls')
    call require(c>0,'passive-aggressive C must be positive')
    call require(tol>=0.or.tol== -1,'passive-aggressive tol must be nonnegative or -1')
    call require(max_iter>0.and.patience>0,'invalid passive-aggressive iteration controls')
    call require(average>=0,'passive-aggressive average must be nonnegative')
    options%c=c; options%tol=tol; options%max_iter=max_iter; options%n_iter_no_change=patience
    options%average=average; options%shuffle=shuffle; options%fit_intercept=fit_intercept
  end function

  subroutine pa_run(x,y,options,weight_pos,weight_neg,coef,intercept,n_iter,converged,last_loss)
    real(dp), intent(in) :: x(:,:),y(:),weight_pos,weight_neg
    type(pa_config), intent(in) :: options
    real(dp), intent(out) :: coef(:),intercept,last_loss
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp) :: current(size(coef)),average_coef(size(coef)),bias,average_bias,prediction,loss,update,norm
    real(dp) :: sumloss,best_loss,coin,seen,average_count
    integer :: order(size(y)),i,j,it,pick,tmp,stale,n
    n=size(y); current=0; average_coef=0; bias=0; average_bias=0
    best_loss=huge(1._dp); stale=0; seen=1; converged=.false.; order=[(i,i=1,n)]
    do it=1,options%max_iter
      if(options%shuffle) then
        do j=n,2,-1
          call random_number(coin); pick=min(j,1+int(coin*j))
          tmp=order(j); order(j)=order(pick); order(pick)=tmp
        end do
      end if
      sumloss=0
      do j=1,n
        i=order(j); prediction=dot_product(current,x(i,:))+bias
        if(options%classification) then
          loss=max(0._dp,1-y(i)*prediction)
        else
          loss=max(0._dp,abs(y(i)-prediction)-options%epsilon)
        end if
        sumloss=sumloss+loss; norm=sum(x(i,:)**2)
        if(options%pa_two) then
          update=loss/(norm+0.5_dp/options%c)
        else
          ! sklearn PA-I skips zero-norm rows, including intercept/averaging.
          if(norm==0) cycle
          update=min(options%c,loss/norm)
        end if
        if(options%classification) then
          update=update*y(i)
        else
          if(y(i)-prediction<0) update=-update
        end if
        if(y(i)>0) then
          update=update*weight_pos
        else
          update=update*weight_neg
        end if
        current=current+update*x(i,:)
        if(options%fit_intercept) bias=bias+update
        if(options%average>0.and.seen>=options%average) then
          average_count=seen-options%average+1
          average_coef=average_coef+(current-average_coef)/average_count
          average_bias=average_bias+(bias-average_bias)/average_count
        end if
        seen=seen+1
      end do
      call require(all(ieee_is_finite(current)).and.ieee_is_finite(bias), &
        'passive-aggressive fit overflow; scale the inputs')
      if(options%tol>=0) then
        if(sumloss>best_loss-options%tol*n) then
          stale=stale+1
        else
          stale=0
        end if
      end if
      best_loss=min(best_loss,sumloss)
      if(stale>=options%n_iter_no_change) then
        converged=.true.; exit
      end if
    end do
    n_iter=min(it,options%max_iter); last_loss=sumloss/n; coef=current; intercept=bias
    ! The public averaging selection uses epoch sample counts, as sklearn does.
    if(options%average>0.and.real(options%average,dp)<=real(n_iter,dp)*n) then
      coef=average_coef; intercept=average_bias
    end if
  end subroutine

  subroutine pa_regressor_fit(self,x,y)
    class(passive_aggressive_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(pa_config) :: options
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    options=pa_settings(self%c,self%tol,self%max_iter,self%n_iter_no_change,self%average,self%shuffle,self%fit_intercept)
    call require(ieee_is_finite(self%epsilon),'nonfinite passive-aggressive epsilon')
    call require(self%epsilon>=0,'passive-aggressive epsilon must be nonnegative')
    call require(self%loss=='epsilon_insensitive'.or.self%loss=='squared_epsilon_insensitive', &
      'invalid passive-aggressive regression loss')
    options%epsilon=self%epsilon; options%pa_two=self%loss=='squared_epsilon_insensitive'
    if(self%shuffle) call seed_rng(self%random_state)
    self%coef_=spread(0._dp,1,size(x,2))
    call pa_run(x,y,options,1._dp,1._dp,self%coef_,self%intercept_,self%n_iter_,self%converged_,self%loss_)
    self%t_=1+real(self%n_iter_,dp)*size(y); self%rank_=0
  end subroutine

  subroutine pa_classifier_fit(self,x,y)
    class(passive_aggressive_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(pa_config) :: options
    real(dp) :: target(size(y)),positive_weight,negative_weight
    real(dp), allocatable :: weights(:)
    integer :: k,heads,j,positive
    logical :: converged
    call check_xy(x,size(y))
    options=pa_settings(self%c,self%tol,self%max_iter,self%n_iter_no_change,self%average,self%shuffle,self%fit_intercept)
    call require(self%loss=='hinge'.or.self%loss=='squared_hinge','invalid passive-aggressive classification loss')
    self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'passive-aggressive classification requires at least two classes')
    heads=k; if(k==2) heads=1
    if(allocated(self%coef_)) deallocate(self%coef_,self%intercept_,self%n_iter_per_class_,self%loss_)
    allocate(self%coef_(heads,size(x,2)),self%intercept_(heads),self%n_iter_per_class_(heads),self%loss_(heads))
    options%classification=.true.; options%pa_two=self%loss=='squared_hinge'
    weights=spread(1._dp,1,k)
    if(self%class_weight_balanced) then
      do j=1,k
        weights(j)=real(size(y),dp)/(real(k,dp)*count(y==self%classes_(j)))
      end do
    end if
    if(self%shuffle) call seed_rng(self%random_state)
    self%converged_=.true.
    do j=1,heads
      positive=j; if(k==2) positive=2
      target=merge(1._dp,-1._dp,y==self%classes_(positive))
      positive_weight=weights(positive); negative_weight=1
      if(k==2) negative_weight=weights(1)
      call pa_run(x,target,options,positive_weight,negative_weight,self%coef_(j,:),self%intercept_(j), &
        self%n_iter_per_class_(j),converged,self%loss_(j))
      self%converged_=self%converged_.and.converged
    end do
    self%n_iter_=maxval(self%n_iter_per_class_); self%t_=1+real(self%n_iter_,dp)*size(y)
    if(self%average>0.and.real(self%average,dp)<=self%t_-1) then
      do j=1,heads
        if(real(self%average,dp)>real(self%n_iter_per_class_(j),dp)*size(y)) then
          self%coef_(j,:)=0; self%intercept_(j)=0
        end if
      end do
    end if
  end subroutine

  function pa_classifier_decision(self,x) result(z)
    class(passive_aggressive_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'passive-aggressive classifier is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  function pa_classifier_predict(self,x) result(y)
    class(passive_aggressive_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=merge(self%classes_(2),self%classes_(1),z(:,1)>0)
    else
      do i=1,size(y)
        y(i)=self%classes_(maxloc(z(i,:),dim=1))
      end do
    end if
  end function

  real(dp) function pa_classifier_score(self,x,y) result(value)
    class(passive_aggressive_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function

  ! sklearn 1.6.1 linear_model/_theil_sen.py, BSD-3-Clause.
  ! See LICENSES/scikit-learn-COPYING and THIRD_PARTY_NOTICES.md.
  subroutine theil_spatial_median(weights,max_iter,tol,center,n_iter,converged)
    real(dp), intent(in) :: weights(:,:),tol
    integer, intent(in) :: max_iter
    real(dp), intent(out) :: center(:)
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp) :: old(size(center)),direction(size(center)),quotient(size(center)),diff(size(center))
    real(dp) :: norm,quotient_norm,denominator,mix
    integer :: i,it
    logical :: coincident
    if(size(center)==1) then
      center(1)=sample_median(weights(:,1)); n_iter=1; converged=.true.; return
    end if
    old=sum(weights,dim=1)/size(weights,1); converged=.false.
    do it=1,max_iter
      quotient=0; direction=0; denominator=0; coincident=.false.
      do i=1,size(weights,1)
        diff=weights(i,:)-old; norm=sqrt(sum(diff**2))
        if(norm>=epsilon(1._dp)) then
          quotient=quotient+diff/norm
          direction=direction+weights(i,:)/norm; denominator=denominator+1/norm
        else
          coincident=.true.
        end if
      end do
      quotient_norm=sqrt(sum(quotient**2))
      if(quotient_norm>epsilon(1._dp)) then
        direction=direction/denominator
      else
        ! Preserve sklearn 1.6.1's modified-Weiszfeld zero-quotient convention.
        direction=1; quotient_norm=1
      end if
      mix=0
      if(coincident) mix=min(1._dp,1/quotient_norm)
      center=(1-mix)*direction+mix*old
      n_iter=it-1 ! sklearn reports the zero-based last Weiszfeld iteration.
      if(sum((center-old)**2)<tol**2) then
        converged=.true.; exit
      end if
      old=center
    end do
  end subroutine

  subroutine theil_sen_fit(self,x,y)
    class(theil_sen_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: design(:,:),weights(:,:),s(:),u(:,:),v(:,:),center(:)
    real(dp) :: combinations,coin,cutoff
    integer :: n,p,q,m,k,i,j,h,pick,tmp,first
    integer, allocatable :: subset(:),pool(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%tol),'nonfinite TheilSen tolerance')
    call require(self%tol>0.and.self%max_iter>0,'invalid TheilSen iteration controls')
    call require(self%max_subpopulation>0,'TheilSen max_subpopulation must be positive')
    call require(self%n_subsamples>=0,'invalid TheilSen n_subsamples')
    n=size(x,1); p=size(x,2); q=p+merge(1,0,self%fit_intercept); m=self%n_subsamples
    if(m==0) then
      m=min(q,n)
    else
      call require(m<=n,'TheilSen n_subsamples exceeds sample count')
      if(n>=p) then
        call require(m>=q,'TheilSen n_subsamples is smaller than design dimension')
      else
        call require(m==n,'underdetermined TheilSen requires all samples')
      end if
    end if
    self%n_subsamples_=m
    self%breakdown_=1-(0.5_dp**(1._dp/m)*(n-m+1)+m-1)/n
    ! Bound the binomial count before integer conversion to avoid overflow.
    combinations=1
    do j=1,min(m,n-m)
      combinations=combinations*real(n-min(m,n-m)+j,dp)/j
      if(combinations>real(self%max_subpopulation,dp)+0.5_dp) exit
    end do
    self%exhaustive_=combinations<=real(self%max_subpopulation,dp)+0.5_dp
    self%n_subpopulation_=self%max_subpopulation
    if(self%exhaustive_) self%n_subpopulation_=max(1,nint(combinations))
    if(allocated(self%subsets_)) deallocate(self%subsets_)
    allocate(self%subsets_(self%n_subpopulation_,m),subset(m),pool(n))
    if(self%exhaustive_) then
      subset=[(i,i=1,m)]
      do i=1,self%n_subpopulation_
        self%subsets_(i,:)=subset
        if(i==self%n_subpopulation_) exit
        j=m
        do while(j>0)
          if(subset(j)<n-m+j) exit
          j=j-1
        end do
        call require(j>0,'TheilSen combination enumeration failed')
        subset(j)=subset(j)+1
        do h=j+1,m
          subset(h)=subset(h-1)+1
        end do
      end do
    else
      call seed_rng(self%random_state)
      do i=1,self%n_subpopulation_
        pool=[(j,j=1,n)]
        do j=1,m
          call random_number(coin); pick=min(n,j+int(coin*(n-j+1)))
          tmp=pool(j); pool(j)=pool(pick); pool(pick)=tmp
        end do
        self%subsets_(i,:)=pool(:m)
      end do
    end if
    allocate(design(m,q),weights(self%n_subpopulation_,q),s(q),u(m,q),v(q,q),center(q))
    first=merge(2,1,self%fit_intercept)
    do i=1,self%n_subpopulation_
      subset=self%subsets_(i,:); design(:,first:)=x(subset,:)
      if(self%fit_intercept) design(:,1)=1
      call dense_svd(design,s,u,v); weights(i,:)=0
      cutoff=epsilon(1._dp)*max(m,q)*maxval(s)
      do k=1,q
        if(s(k)>cutoff) weights(i,:)=weights(i,:)+v(:,k)*dot_product(u(:,k),y(subset))/s(k)
      end do
    end do
    call theil_spatial_median(weights,self%max_iter,self%tol,center,self%n_iter_,self%converged_)
    self%intercept_=0; self%coef_=center(first:); self%rank_=0
    if(self%fit_intercept) self%intercept_=center(1)
  end subroutine

  ! sklearn 1.6.1 _huber.py, _ransac.py and _glm/glm.py are the references
  ! for these objectives and estimator contracts. See THIRD_PARTY_NOTICES.md.
  subroutine huber_objective(design,y,w,p,epsilon,alpha,loss,gradient)
    real(dp), intent(in) :: design(:,:),y(:),w(:),epsilon,alpha
    integer, intent(in) :: p
    real(dp), intent(out) :: loss,gradient(:)
    real(dp) :: r(size(y)),force(size(y)),sigma,quadratic
    logical :: out(size(y))
    integer :: q,n
    n=size(y); q=size(w); sigma=w(q)
    r=y-matmul(design,w(:q-1)); out=abs(r)>epsilon*sigma
    force=0; where(out) force=2*epsilon*sign(1._dp,r)
    where(.not.out) force=2*r/sigma
    quadratic=sum(pack(r,.not.out)**2)/sigma
    loss=(n*sigma+quadratic+2*epsilon*sum(pack(abs(r),out))-count(out)*sigma*epsilon**2 &
      +alpha*sum(w(:p)**2))/n
    gradient(:q-1)=-matmul(transpose(design),force)/n
    gradient(:p)=gradient(:p)+2*alpha*w(:p)/n
    gradient(q)=(n-count(out)*epsilon**2-quadratic/sigma)/n
  end subroutine
  subroutine huber_fit(self,x,y)
    class(huber_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: design(:,:),w(:),trial(:),g(:),gt(:),pg(:),h(:,:),direction(:),s(:),v(:),hv(:)
    real(dp) :: loss,next,rate,sy,lower,slope
    integer :: n,p,q,j,it,ls
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%epsilon>=1.and.self%alpha>=0.and.self%tol>0.and.self%max_iter>0, &
      'invalid HuberRegressor controls')
    n=size(y); p=size(x,2); q=p+1
    if(self%fit_intercept) q=q+1
    allocate(design(n,q-1),w(q),trial(q),g(q),gt(q),pg(q),h(q,q),direction(q),s(q),v(q),hv(q))
    design(:,1:p)=x; if(self%fit_intercept) design(:,p+1)=1
    lower=10*epsilon(1._dp); w=0; w(q)=1; h=0
    do j=1,q
      h(j,j)=1
    end do
    call huber_objective(design,y,w,p,self%epsilon,self%alpha,loss,g)
    self%converged_=.false.; self%n_iter_=0
    do it=1,self%max_iter
      pg=g
      if(w(q)<=lower.and.g(q)>0) pg(q)=0
      if(maxval(abs(pg))*n<=self%tol) then
        self%converged_=.true.; exit
      end if
      direction=-matmul(h,pg)
      if(w(q)<=lower.and.direction(q)<0) direction(q)=0
      if(dot_product(g,direction)>=0) direction=-pg
      rate=1
      do ls=1,80
        trial=w+rate*direction; trial(q)=max(lower,trial(q))
        slope=dot_product(g,trial-w)
        call huber_objective(design,y,trial,p,self%epsilon,self%alpha,next,gt)
        if(slope<0.and.next<=loss+1.e-4_dp*slope) exit
        if(slope<0.and.abs(next-loss)<=8*epsilon(1._dp)*max(1._dp,abs(loss)) &
          .and.maxval(abs(gt))<maxval(abs(g))) exit
        rate=rate*0.5_dp
      end do
      if(ls>80) exit
      s=trial-w; v=gt-g; sy=dot_product(s,v)
      if(sy>epsilon(1._dp)*sqrt(sum(s*s)*sum(v*v))) then
        hv=matmul(h,v)
        h=h+(1+dot_product(v,hv)/sy)*spread(s,2,q)*spread(s,1,q)/sy &
          -(spread(hv,2,q)*spread(s,1,q)+spread(s,2,q)*spread(hv,1,q))/sy
      end if
      w=trial; loss=next; g=gt; self%n_iter_=it
    end do
    pg=g; if(w(q)<=lower.and.g(q)>0) pg(q)=0
    if(maxval(abs(pg))*n<=self%tol) self%converged_=.true.
    self%coef_=w(:p); self%intercept_=0
    if(self%fit_intercept) self%intercept_=w(p+1)
    self%scale_=w(q); self%loss_=loss*n
    self%outliers_=abs(y-self%predict(x))>self%epsilon*self%scale_
  end subroutine
  real(dp) function sample_median(values) result(median)
    real(dp), intent(in) :: values(:)
    integer :: order(size(values)),n
    n=size(values); order=argsort(values)
    median=0.5_dp*(values(order((n+1)/2))+values(order(n/2+1)))
  end function
  subroutine ransac_fit(self,x,y)
    class(ransac_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(linear_regression) :: candidate
    integer :: n,p,m,j,k,t,best_count,n_inliers,cap
    integer, allocatable :: indices(:),inliers(:)
    real(dp) :: threshold,coin,score,best_score,den,bound,residual(size(y))
    logical :: mask(size(y))
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    n=size(y); p=size(x,2); m=self%min_samples
    if(m==0) m=p+1
    call require(m>=1.and.m<=n.and.self%max_trials>0,'invalid RANSAC sample/trial controls')
    call require(self%stop_probability>=0.and.self%stop_probability<=1.and.self%stop_n_inliers>=1, &
      'invalid RANSAC stopping controls')
    call require(self%residual_threshold>=0.or.self%residual_threshold==-1,'invalid RANSAC residual threshold')
    call require(self%loss=='absolute_error'.or.self%loss=='squared_error','invalid RANSAC loss')
    threshold=self%residual_threshold
    if(threshold==-1) threshold=sample_median(abs(y-sample_median(y)))
    cap=self%max_trials; best_count=0; best_score=-huge(1._dp)
    self%n_trials_=0; self%n_skips_no_inliers_=0; self%inlier_mask_=spread(.false.,1,n)
    candidate%fit_intercept=self%fit_intercept
    call seed_rng(self%random_state)
    do while(self%n_trials_<cap)
      self%n_trials_=self%n_trials_+1; indices=[(j,j=1,n)]
      do j=1,m
        call random_number(coin); k=j+int(coin*(n-j+1))
        t=indices(j); indices(j)=indices(k); indices(k)=t
      end do
      call candidate%fit(x(indices(:m),:),y(indices(:m)))
      residual=abs(y-candidate%predict(x))
      if(self%loss=='squared_error') residual=residual**2
      mask=residual<=threshold; n_inliers=count(mask)
      if(n_inliers<max(1,best_count)) then
        self%n_skips_no_inliers_=self%n_skips_no_inliers_+1; cycle
      end if
      inliers=pack([(j,j=1,n)],mask); score=-huge(1._dp)
      if(n_inliers>1) score=candidate%score(x(inliers,:),y(inliers))
      if(n_inliers==best_count.and.score<best_score) cycle
      best_count=n_inliers; best_score=score; self%inlier_mask_=mask
      den=max(epsilon(1._dp),1-(real(n_inliers,dp)/n)**m)
      if(den<1) then
        bound=log(max(epsilon(1._dp),1-self%stop_probability))/log(den)
        cap=ceiling(min(real(cap,dp),max(1._dp,bound)))
      end if
      if(n_inliers>=self%stop_n_inliers.or.score>=self%stop_score) exit
    end do
    call require(best_count>0,'RANSAC could not find a consensus set')
    inliers=pack([(j,j=1,n)],self%inlier_mask_)
    call candidate%fit(x(inliers,:),y(inliers))
    self%coef_=candidate%coef_; self%intercept_=candidate%intercept_; self%rank_=candidate%rank_
  end subroutine
  subroutine poisson_fit(self,x,y)
    class(poisson_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%power=1; self%link='log'; call tweedie_fit(self,x,y)
  end subroutine
  subroutine gamma_fit(self,x,y)
    class(gamma_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%power=2; self%link='log'; call tweedie_fit(self,x,y)
  end subroutine
  subroutine glm_objective(design,y,w,p,power,link,alpha,loss,g,h,valid)
    real(dp), intent(in) :: design(:,:),y(:),w(:),power,alpha
    character(*), intent(in) :: link
    integer, intent(in) :: p
    real(dp), intent(out) :: loss,g(:),h(:,:)
    logical, intent(out) :: valid
    real(dp) :: eta(size(y)),mu(size(y)),gradient(size(y)),curvature(size(y)),a(size(y)),b(size(y))
    integer :: n,j
    n=size(y); eta=matmul(design,w); valid=.true.; g=0; h=0; loss=huge(1._dp)
    if(link=='log') then
      valid=maxval(abs(eta))<0.25_dp*log(huge(1._dp))
      if(.not.valid) return
      mu=exp(eta)
    else
      mu=eta
    end if
    if(power==0) then
      loss=0.5_dp*sum((mu-y)**2)/n; gradient=mu-y; curvature=1
    else if(power==1) then
      loss=sum(mu-y*eta)/n; gradient=mu-y; curvature=mu
    else if(power==2) then
      a=y/mu; loss=sum(a+eta)/n; gradient=1-a; curvature=a
    else
      a=exp((2-power)*eta); b=y*exp((1-power)*eta)
      ! Subtract eta-independent constants and avoid cancellation near p=1/2.
      loss=sum(exponential_minus_one((2-power)*eta)/(2-power) &
        -y*exponential_minus_one((1-power)*eta)/(1-power))/n
      gradient=a-b; curvature=(2-power)*a+(power-1)*b
    end if
    loss=loss+0.5_dp*alpha*sum(w(:p)**2)
    g=matmul(transpose(design),gradient)/n; g(:p)=g(:p)+alpha*w(:p)
    h=matmul(transpose(design),design*spread(curvature,2,size(w)))/n
    do j=1,p
      h(j,j)=h(j,j)+alpha
    end do
  end subroutine
  elemental real(dp) function exponential_minus_one(z) result(value)
    real(dp), intent(in) :: z
    if(abs(z)<1.e-4_dp) then
      value=z*(1+z*(0.5_dp+z*(1._dp/6+z*(1._dp/24+z*(1._dp/120+z/720)))))
    else
      value=exp(z)-1
    end if
  end function
  subroutine tweedie_fit(self,x,y)
    class(tweedie_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: design(:,:),w(:),trial(:),g(:),h(:,:),gt(:),ht(:,:),step(:),s(:),v(:,:)
    real(dp) :: loss,next,rate,cutoff
    integer :: n,p,q,it,ls
    logical :: valid
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%power==0.or.(self%power>=1.and.self%power<=2),'supported Tweedie powers are 0 or [1,2]')
    call require(self%alpha>=0.and.self%tol>0.and.self%max_iter>0,'invalid GLM controls')
    self%link_=self%link
    if(self%link=='auto') self%link_=merge('identity','log     ',self%power==0)
    call require((self%power==0.and.self%link_=='identity').or.(self%power>=1.and.self%link_=='log'), &
      'supported GLM links are identity for power 0 and log for powers [1,2]')
    if(self%power>0.and.self%power<2) call require(all(y>=0),'GLM targets must be nonnegative')
    if(self%power==2) call require(all(y>0),'Gamma targets must be positive')
    if(self%link_=='log') call require(sum(y)>0,'log-link GLM requires a positive target mean')
    n=size(y); p=size(x,2); q=p
    if(self%fit_intercept) q=q+1
    allocate(design(n,q),w(q),trial(q),g(q),h(q,q),gt(q),ht(q,q),step(q),s(q),v(q,q))
    design(:,:p)=x; if(self%fit_intercept) design(:,q)=1
    w=0
    if(self%fit_intercept) then
      w(q)=sum(y)/n; if(self%link_=='log') w(q)=log(w(q))
    end if
    call glm_objective(design,y,w,p,self%power,self%link_,self%alpha,loss,g,h,valid)
    call require(valid,'GLM initial prediction exceeds numerical range')
    self%converged_=.false.; self%n_iter_=0
    do it=1,self%max_iter
      if(maxval(abs(g))<=self%tol) then
        self%converged_=.true.; exit
      end if
      call symmetric_eigen(h,s,v); cutoff=epsilon(1._dp)*q*maxval(s)
      step=0
      where(s>cutoff) step=matmul(transpose(v),g)/max(s,tiny(1._dp))
      step=matmul(v,step); rate=1
      do ls=1,80
        trial=w-rate*step
        call glm_objective(design,y,trial,p,self%power,self%link_,self%alpha,next,gt,ht,valid)
        if(valid.and.next<=loss-1.e-4_dp*rate*dot_product(g,step)) exit
        if(valid.and.abs(next-loss)<=8*epsilon(1._dp)*max(1._dp,abs(loss)) &
          .and.maxval(abs(gt))<maxval(abs(g))) exit
        rate=rate*0.5_dp
      end do
      if(ls>80) exit
      w=trial; g=gt; h=ht; loss=next; self%n_iter_=it
    end do
    if(maxval(abs(g))<=self%tol) self%converged_=.true.
    self%coef_=w(:p); self%intercept_=0
    if(self%fit_intercept) self%intercept_=w(q)
  end subroutine
  function glm_predict(self,x) result(y)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%coef_),'GLM is not fitted')
    call check_xy(transpose(x),size(self%coef_)); y=matmul(x,self%coef_)+self%intercept_
    if(self%link_=='log') then
      call require(maxval(abs(y))<log(huge(1._dp)),'GLM prediction exceeds numerical range')
      y=exp(y)
    end if
  end function
  real(dp) function tweedie_deviance(y,mu,power) result(d)
    real(dp), intent(in) :: y(:),mu(:),power
    real(dp) :: term(size(y)),a,b,ratio,log_ratio
    integer :: i
    if(power==0) then
      d=sum((y-mu)**2)/size(y)
    else if(power==1) then
      term=mu-y
      do i=1,size(y)
        if(y(i)>0) term(i)=term(i)+y(i)*log(y(i)/mu(i))
      end do
      d=2*sum(term)/size(y)
    else if(power==2) then
      d=2*sum(log(mu/y)+y/mu-1)/size(y)
    else
      a=2-power; b=1-power
      do i=1,size(y)
        if(y(i)==0) then
          term(i)=mu(i)**a/a
        else
          ratio=y(i)/mu(i); log_ratio=log(ratio)
          if(power<=1.5_dp) then
            term(i)=mu(i)**a*(ratio*exponential_minus_one(b*log_ratio)/b-(ratio-1))/a
          else
            term(i)=mu(i)**a*(exponential_minus_one(a*log_ratio)/a-(ratio-1))/b
          end if
        end if
      end do
      d=2*sum(term)/size(y)
    end if
  end function
  real(dp) function glm_score(self,x,y) result(s)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: null_deviance,mu(size(y))
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    if(self%power>0.and.self%power<2) call require(all(y>=0),'GLM targets must be nonnegative')
    if(self%power==2) call require(all(y>0),'Gamma targets must be positive')
    call require(maxval(y)>minval(y),'D2 requires a nonconstant target')
    mu=sum(y)/size(y); null_deviance=tweedie_deviance(y,mu,self%power)
    s=1-tweedie_deviance(y,self%predict(x),self%power)/null_deviance
  end function
  ! Bayesian precision updates and OMP selection follow sklearn 1.6.1
  ! linear_model/_bayes.py and _omp.py. See THIRD_PARTY_NOTICES.md.
  subroutine bayesian_fit(self,x,y)
    class(bayesian_ridge), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),s(size(x,2)),v(size(x,2),size(x,2))
    real(dp) :: rhs(size(x,2)),old(size(x,2)),weights(size(x,2)),gamma,rss
    integer :: it,j,p,k
    call require(self%max_iter>0.and.self%tol>=0,'invalid BayesianRidge iteration parameters')
    call require(min(self%alpha_1,self%alpha_2,self%lambda_1,self%lambda_2)>0, &
      'BayesianRidge hyperpriors must be positive')
    call require((self%alpha_init==-1.or.self%alpha_init>0).and.self%lambda_init>0, &
      'invalid BayesianRidge initial precision')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    p=size(x,2); k=min(size(x,1),p)
    call dense_svd(xc,s,u,v); rhs=s*matmul(transpose(u),yc)
    self%rank_=count(s>epsilon(1._dp)*maxval(s)*max(size(x,1),p))
    self%alpha_=self%alpha_init
    if(self%alpha_init==-1) self%alpha_=1/(sum((yc-sum(yc)/size(yc))**2)/size(yc)+epsilon(1._dp))
    self%lambda_=self%lambda_init; old=0; self%converged_=.false.
    do it=1,self%max_iter
      self%coef_=matmul(v,rhs/(s*s+self%lambda_/self%alpha_))
      rss=sum((yc-matmul(xc,self%coef_))**2)
      gamma=sum(self%alpha_*s*s/(self%lambda_+self%alpha_*s*s))
      self%lambda_=(gamma+2*self%lambda_1)/(sum(self%coef_**2)+2*self%lambda_2)
      self%alpha_=(size(y)-gamma+2*self%alpha_1)/(rss+2*self%alpha_2)
      self%n_iter_=it
      if(it>1.and.sum(abs(old-self%coef_))<self%tol) then
        self%converged_=.true.; exit
      end if
      old=self%coef_
    end do
    self%coef_=matmul(v,rhs/(s*s+self%lambda_/self%alpha_))
    self%intercept_=ym-dot_product(xm,self%coef_)
    weights=0; weights(:k)=1/(self%alpha_*s(:k)**2+self%lambda_)
    self%sigma_=matmul(v*spread(weights,1,p),transpose(v))
  end subroutine
  function bayesian_std(self,x) result(std)
    class(bayesian_ridge), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    call require(allocated(self%sigma_),'BayesianRidge is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    std=sqrt(max(0._dp,sum(matmul(x,self%sigma_)*x,dim=2))+1/self%alpha_)
  end function
  function positive_inverse(a) result(inv)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inv(size(a,1),size(a,2)),l(size(a,1),size(a,2)),e(size(a,1))
    integer :: j
    l=cholesky_factor(a)
    do j=1,size(a,1)
      e=0; e(j)=1; inv(:,j)=solve_upper(transpose(l),solve_lower(l,e))
    end do
  end function
  subroutine ard_posterior(x,y,alpha,lambda,keep,sigma,coef)
    real(dp), intent(in) :: x(:,:),y(:),alpha,lambda(:)
    integer, intent(in) :: keep(:)
    real(dp), allocatable, intent(out) :: sigma(:,:)
    real(dp), intent(out) :: coef(:)
    real(dp) :: a(size(keep),size(keep))
    integer :: j
    coef=0
    a=alpha*matmul(transpose(x(:,keep)),x(:,keep))
    do j=1,size(keep)
      a(j,j)=a(j,j)+lambda(keep(j))
    end do
    sigma=positive_inverse(a)
    coef(keep)=alpha*matmul(sigma,matmul(transpose(x(:,keep)),y))
  end subroutine
  subroutine ard_fit(self,x,y)
    class(ard_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: old(size(x,2)),gamma(size(x,2)),rss
    integer, allocatable :: keep(:)
    integer :: it,j,p
    call require(self%max_iter>0.and.self%tol>=0.and.self%threshold_lambda>0,'invalid ARD parameters')
    call require(min(self%alpha_1,self%alpha_2,self%lambda_1,self%lambda_2)>0,'ARD hyperpriors must be positive')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); p=size(x,2)
    self%lambda_=spread(1._dp,1,p); self%coef_=spread(0._dp,1,p)
    self%alpha_=1/(sum((yc-sum(yc)/size(yc))**2)/size(yc)+epsilon(1._dp))
    keep=[(j,j=1,p)]; old=0; self%converged_=.false.
    do it=1,self%max_iter
      call ard_posterior(xc,yc,self%alpha_,self%lambda_,keep,self%sigma_,self%coef_)
      rss=sum((yc-matmul(xc,self%coef_))**2)
      do j=1,size(keep)
        gamma(j)=1-self%lambda_(keep(j))*self%sigma_(j,j)
      end do
      self%lambda_(keep)=(gamma(:size(keep))+2*self%lambda_1)/(self%coef_(keep)**2+2*self%lambda_2)
      self%alpha_=(size(y)-sum(gamma(:size(keep)))+2*self%alpha_1)/(rss+2*self%alpha_2)
      keep=pack([(j,j=1,p)],self%lambda_<self%threshold_lambda)
      where(self%lambda_>=self%threshold_lambda) self%coef_=0
      self%n_iter_=it
      if((it>1.and.sum(abs(old-self%coef_))<self%tol).or.size(keep)==0) then
        self%converged_=.true.; exit
      end if
      old=self%coef_
    end do
    call ard_posterior(xc,yc,self%alpha_,self%lambda_,keep,self%sigma_,self%coef_)
    self%intercept_=ym-dot_product(xm,self%coef_); self%rank_=size(keep)
  end subroutine
  function ard_std(self,x) result(std)
    class(ard_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    integer, allocatable :: keep(:)
    integer :: j
    call require(allocated(self%sigma_),'ARDRegression is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    keep=pack([(j,j=1,size(self%coef_))],self%lambda_<self%threshold_lambda)
    std=sqrt(max(0._dp,sum(matmul(x(:,keep),self%sigma_)*x(:,keep),dim=2))+1/self%alpha_)
  end function
  subroutine omp_fit(self,x,y)
    class(orthogonal_matching_pursuit), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: residual(size(y)),corr(size(x,2))
    real(dp), allocatable :: s(:),u(:,:),v(:,:),c(:)
    integer :: p,k,j,limit,idx(size(x,2))
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); p=size(x,2)
    call require(self%n_nonzero_coefs>=0.and.self%n_nonzero_coefs<=p,'invalid OMP sparsity')
    call require(self%tol>=0.or.self%tol==-1,'OMP tol must be nonnegative or -1')
    limit=self%n_nonzero_coefs
    if(limit==0) limit=max(1,p/10)
    if(self%tol>=0) limit=p
    residual=yc; self%coef_=spread(0._dp,1,p); self%n_iter_=0
    do k=1,limit
      corr=matmul(transpose(xc),residual); j=maxloc(abs(corr),dim=1)
      if(any(idx(:k-1)==j).or.corr(j)**2<epsilon(1._dp)) exit
      idx(k)=j
      if(allocated(s)) deallocate(s,u,v,c)
      allocate(s(k),u(size(y),k),v(k,k),c(k))
      call dense_svd(xc(:,idx(:k)),s,u,v)
      if(minval(s)<=sqrt(epsilon(1._dp))*maxval(s)) exit
      c=matmul(v,matmul(transpose(u),yc)/s)
      self%coef_=0; self%coef_(idx(:k))=c
      residual=yc-matmul(xc,self%coef_); self%n_iter_=k
      if(self%tol>=0.and.sum(residual**2)<=self%tol) exit
    end do
    self%rank_=self%n_iter_; self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine centered(x,y,fit_intercept,xc,yc,xmean,ymean)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in) :: fit_intercept
    real(dp), intent(out) :: xc(:,:),yc(:),xmean(:),ymean
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    xmean=0; ymean=0
    if(fit_intercept) then
      xmean=sum(x,dim=1)/size(x,1); ymean=sum(y)/size(y)
    end if
    xc=x-spread(xmean,1,size(x,1)); yc=y-ymean
  end subroutine
  subroutine spectral_fit(self,x,y,alpha)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),alpha
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),vals(size(x,2)),vecs(size(x,2),size(x,2))
    real(dp) :: rhs(size(x,2)),projection(size(x,2)),cutoff
    integer :: j,p
    call require(alpha>=0,'alpha must be nonnegative'); p=size(x,2)
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    call dense_svd(xc,vals,u,vecs)
    cutoff=epsilon(1._dp)*max(size(x,1),p)*maxval(vals)
    self%rank_=count(vals>cutoff); projection=matmul(transpose(u),yc)
    do j=1,p
      if(vals(j)>cutoff.or.alpha>0) then
        projection(j)=projection(j)*vals(j)/(vals(j)**2+alpha)
      else
        projection(j)=0
      end if
    end do
    self%coef_=matmul(vecs,projection); self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine ols_fit(self,x,y)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call spectral_fit(self,x,y,0._dp)
  end subroutine
  subroutine ridge_fit(self,x,y)
    class(ridge), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call spectral_fit(self,x,y,self%alpha)
  end subroutine
  function linear_predict(self,x) result(y)
    class(linear_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%coef_),'linear model is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function linear_score(self,x,y) result(s)
    class(linear_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    s=r2_score(y,self%predict(x))
  end function
  subroutine lasso_fit(self,x,y)
    class(lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%l1_ratio=1
    call elastic_fit(self,x,y)
  end subroutine
  subroutine elastic_fit(self,x,y)
    class(elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: residual(size(y)),old,update,rho,den,change
    integer :: it,j,n
    call require(self%alpha>=0.and.self%l1_ratio>=0.and.self%l1_ratio<=1,'invalid elastic net penalties')
    call require(self%max_iter>0.and.self%tol>0,'invalid elastic net iteration controls')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); n=size(y)
    self%coef_=spread(0._dp,1,size(x,2)); residual=yc; self%converged_=.false.
    do it=1,self%max_iter
      change=0
      do j=1,size(x,2)
        old=self%coef_(j); residual=residual+xc(:,j)*old
        rho=dot_product(xc(:,j),residual)/n
        den=sum(xc(:,j)**2)/n+self%alpha*(1-self%l1_ratio)
        update=0
        if(den>0) update=sign(max(abs(rho)-self%alpha*self%l1_ratio,0._dp),rho)/den
        self%coef_(j)=update; residual=residual-xc(:,j)*update
        change=max(change,abs(update-old))
      end do
      if(change<self%tol*max(1._dp,maxval(abs(self%coef_)))) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter); self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine lasso_cv_fit(self,x,y)
    class(lasso_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%l1_ratios=[1._dp]; self%l1_ratio=1
    call elastic_cv_fit(self,x,y)
  end subroutine

  subroutine elastic_cv_fit(self,x,y)
    class(elastic_net_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(elastic_net) :: model
    real(dp), allocatable :: ratios(:),grid(:),xc(:,:),yc(:),xm(:),prediction(:)
    integer, allocatable :: indices(:),train(:)
    real(dp) :: ym,alpha_max,mse
    integer :: n,p,na,nr,r,a,f,first,last,fold_size,i,best_a,best_r
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); n=size(y); p=size(x,2)
    call require(self%cv>=2.and.self%cv<=n,'invalid coordinate CV fold count')
    call require(ieee_is_finite(self%tol),'nonfinite coordinate CV tolerance')
    call require(self%tol>0.and.self%max_iter>0,'invalid coordinate CV iteration controls')
    ratios=[self%l1_ratio]
    if(allocated(self%l1_ratios)) ratios=self%l1_ratios
    call require(size(ratios)>0,'empty coordinate CV l1 ratios')
    call require(all(ieee_is_finite(ratios)),'nonfinite coordinate CV l1 ratios')
    call require(all(ratios>0).and.all(ratios<=1),'coordinate CV l1 ratios must be in (0,1]')
    nr=size(ratios)
    if(allocated(self%alphas)) then
      call require(size(self%alphas)>0,'empty coordinate CV alpha grid')
      call require(all(ieee_is_finite(self%alphas)),'nonfinite coordinate CV alphas')
      call require(all(self%alphas>0),'coordinate CV alphas must be positive')
      grid=self%alphas(argsort(-self%alphas)); na=size(grid)
    else
      call require(self%n_alphas>0,'coordinate CV n_alphas must be positive')
      call require(ieee_is_finite(self%eps),'nonfinite coordinate CV eps')
      call require(self%eps>0.and.self%eps<=1,'coordinate CV eps must be in (0,1]')
      na=self%n_alphas
    end if
    if(allocated(self%alphas_)) deallocate(self%alphas_,self%mse_path_)
    allocate(self%alphas_(na,nr),self%mse_path_(na,self%cv,nr))
    allocate(xc(n,p),yc(n),xm(p)); call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    do r=1,nr
      if(allocated(self%alphas)) then
        self%alphas_(:,r)=grid
      else
        alpha_max=maxval(abs(matmul(transpose(xc),yc)))/(n*ratios(r))
        if(alpha_max<=1.e-15_dp) then
          self%alphas_(:,r)=1.e-15_dp
        else
          self%alphas_(:,r)=[(alpha_max*exp(log(self%eps)*real(a-1,dp)/max(1,na-1)),a=1,na)]
        end if
      end if
    end do
    indices=[(i,i=1,n)]; self%all_cv_converged_=.true.
    model%fit_intercept=self%fit_intercept; model%max_iter=self%max_iter; model%tol=self%tol
    do r=1,nr
      model%l1_ratio=ratios(r); first=1
      do f=1,self%cv
        fold_size=n/self%cv
        if(f<=mod(n,self%cv)) fold_size=fold_size+1
        last=first+fold_size-1; train=pack(indices,indices<first.or.indices>last)
        do a=1,na
          model%alpha=self%alphas_(a,r); call model%fit(x(train,:),y(train))
          prediction=model%predict(x(first:last,:))
          self%mse_path_(a,f,r)=sum((prediction-y(first:last))**2)/fold_size
          self%all_cv_converged_=self%all_cv_converged_.and.model%converged_
        end do
        first=last+1
      end do
    end do
    self%best_mse_=huge(1._dp); best_a=1; best_r=1
    do r=1,nr
      do a=1,na
        mse=sum(self%mse_path_(a,:,r))/self%cv
        if(mse<self%best_mse_) then
          self%best_mse_=mse; best_a=a; best_r=r
        end if
      end do
    end do
    self%alpha_=self%alphas_(best_a,best_r); self%l1_ratio_=ratios(best_r)
    self%alpha=self%alpha_; self%l1_ratio=self%l1_ratio_
    call elastic_fit(self,x,y); self%n_features_in_=p
  end subroutine

  subroutine logistic_fit(self,x,y,sample_weight)
    class(logistic_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: observation_weight(size(y)),mass,gradient_tol
    real(dp), allocatable :: design(:,:),w(:,:),prob(:,:),gradient(:,:),hess(:,:),step(:),trial(:,:),scaling(:)
    real(dp) :: objective,new_objective,rate,penalty,v
    integer :: n,p,k,q,it,i,j,a,b,ia,ib,target
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'logistic regression requires at least two classes')
    call require(self%c>0.and.self%tol>0.and.self%max_iter>0,'invalid logistic controls')
    n=size(x,1); p=size(x,2); q=p+merge(1,0,self%fit_intercept)
    ! Binary has one logit; multiclass uses one logit per class and softmax.
    if(k==2) k=1
    allocate(design(n,q),w(q,k),prob(n,k),gradient(q,k),hess(q*k,q*k),step(q*k),trial(q,k),scaling(q*k))
    design(:,:p)=x; if(q>p) design(:,q)=1
    observation_weight=1; gradient_tol=self%tol
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'logistic sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'logistic sample weights must be finite')
      call require(all(sample_weight>=0),'logistic sample weights must be nonnegative')
      call require(maxval(sample_weight)<=sqrt(huge(1._dp))/(16*real(n,dp)), &
        'logistic sample weight mass exceeds working range')
      mass=sum(sample_weight)
      call require(mass>0,'logistic sample weights require positive total')
      if(ieee_is_finite(self%c)) then
        call require(log(self%c)+log(mass)>=-log(huge(1._dp)/16), &
          'weighted logistic penalty exceeds working range')
        penalty=0
        if(mass<=1) then
          penalty=1/(self%c*mass)
        else if(self%c<=huge(1._dp)/mass) then
          penalty=1/(self%c*mass)
        end if
      else
        penalty=0
      end if
      observation_weight=sample_weight/mass
      if(mass>=1) gradient_tol=self%tol/mass
    else
      call require(self%c>=1/huge(1._dp),'logistic C is below native working range')
      penalty=1/self%c
    end if
    w=0; self%converged_=.false.
    do it=1,self%max_iter
      call evaluate(w,objective,prob)
      gradient=0; hess=0
      do i=1,n
        target=label_index(self%classes_,y(i))
        do a=1,k
          v=prob(i,a)
          if((k==1.and.target==2).or.(k>1.and.target==a)) v=v-1
          gradient(:,a)=gradient(:,a)+design(i,:)*(v*observation_weight(i))
          do b=1,k
            v=prob(i,a)*(merge(1._dp,0._dp,a==b)-prob(i,b))*observation_weight(i)
            do j=1,q
              ia=(a-1)*q+j
              hess(ia,(b-1)*q+1:b*q)=hess(ia,(b-1)*q+1:b*q)+v*design(i,j)*design(i,:)
            end do
          end do
        end do
      end do
      gradient(:p,:)=gradient(:p,:)+penalty*w(:p,:)
      do a=1,k
        do j=1,q
          ia=(a-1)*q+j
          if(j<=p) hess(ia,ia)=hess(ia,ia)+penalty
          hess(ia,ia)=hess(ia,ia)+1.e-10_dp
        end do
      end do
      if(maxval(abs(gradient))<gradient_tol) then
        self%converged_=.true.; exit
      end if
      if(present(sample_weight)) then
        ! Equilibrate the weighted Newton system, especially for tiny total
        ! weight with an unregularized intercept and a large scaled penalty.
        do j=1,q*k
          scaling(j)=sqrt(hess(j,j))
        end do
        hess=hess/spread(scaling,2,q*k)/spread(scaling,1,q*k)
        step=solve_system(hess,reshape(gradient,[q*k])/scaling)/scaling
      else
        step=solve_system(hess,reshape(gradient,[q*k]))
      end if
      rate=1
      do j=1,50
        trial=w-rate*reshape(step,[q,k]); call evaluate(trial,new_objective,prob)
        if(new_objective<=objective-1.e-4_dp*rate*sum(reshape(gradient,[q*k])*step)) exit
        rate=rate/2
      end do
      w=trial
      if(maxval(abs(rate*step))<self%tol) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter)
    self%coef_=transpose(w(:p,:)); self%intercept_=spread(0._dp,1,k)
    if(q>p) self%intercept_=w(q,:)
    ! Remove the softmax intercept null mode: fold-dependent numerical drift
    ! changes raw scores used by downstream calibration despite identical proba.
    if(k>1) self%intercept_=self%intercept_-sum(self%intercept_)/k
  contains
    subroutine evaluate(weights,loss,probabilities)
      real(dp), intent(in) :: weights(:,:)
      real(dp), intent(out) :: loss,probabilities(:,:)
      real(dp) :: logits(k),normalizer,z
      integer :: row,cl
      loss=0.5_dp*penalty*sum(weights(:p,:)**2)
      do row=1,n
        logits=matmul(design(row,:),weights)
        cl=label_index(self%classes_,y(row))
        if(k==1) then
          z=logits(1); probabilities(row,1)=sigmoid(z)
          loss=loss+observation_weight(row)*max(z,0._dp)+observation_weight(row)*log(1+exp(-abs(z)))
          if(cl==2) loss=loss-observation_weight(row)*z
        else
          normalizer=logsumexp(logits); probabilities(row,:)=exp(logits-normalizer)
          loss=loss+observation_weight(row)*normalizer-observation_weight(row)*logits(cl)
        end if
      end do
    end subroutine
  end subroutine
  function logistic_decision(self,x) result(z)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'logistic_regression is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function logistic_proba(self,x) result(prob)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),z(:,:)
    integer :: i,k
    z=self%decision_function(x); k=size(self%classes_); allocate(prob(size(x,1),k))
    if(k==2) then
      prob(:,2)=sigmoid(z(:,1)); prob(:,1)=1-prob(:,2)
    else
      do i=1,size(x,1)
        prob(i,:)=exp(z(i,:)-logsumexp(z(i,:)))
      end do
    end if
  end function
  function logistic_predict(self,x) result(y)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: decision(:,:)
    decision=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=self%classes_(merge(2,1,decision(:,1)>0))
    else
      do i=1,size(y)
        y(i)=self%classes_(maxloc(decision(i,:),dim=1))
      end do
    end if
  end function
  real(dp) function logistic_score(self,x,y) result(s)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    s=accuracy_score(y,self%predict(x))
  end function
  subroutine perceptron_fit(self,x,y)
    class(perceptron), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: target,margin
    integer :: it,i,c,k,errors,heads,positive
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2.and.self%max_iter>0.and.self%eta0>0,'invalid perceptron controls')
    if(allocated(self%coef_)) deallocate(self%coef_,self%intercept_)
    heads=k; if(k==2) heads=1
    allocate(self%coef_(heads,size(x,2)),self%intercept_(heads)); self%coef_=0; self%intercept_=0
    do it=1,self%max_iter
      errors=0
      do i=1,size(y)
        do c=1,heads
          positive=c; if(k==2) positive=2
          target=merge(1._dp,-1._dp,y(i)==self%classes_(positive))
          margin=dot_product(self%coef_(c,:),x(i,:))+self%intercept_(c)
          if(target*margin<=0) then
            self%coef_(c,:)=self%coef_(c,:)+self%eta0*target*x(i,:)
            if(self%fit_intercept) self%intercept_(c)=self%intercept_(c)+self%eta0*target
            errors=errors+1
          end if
        end do
      end do
      if(errors==0) exit
    end do
    self%n_iter_=min(it,self%max_iter)
  end subroutine
  function perceptron_decision(self,x) result(z)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'perceptron is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function perceptron_predict(self,x) result(y)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x)
    do i=1,size(y)
      if(size(self%classes_)==2) then
        y(i)=self%classes_(merge(2,1,z(i,1)>0))
      else
        y(i)=self%classes_(maxloc(z(i,:),dim=1))
      end if
    end do
  end function
  real(dp) function perceptron_score(self,x,y) result(s)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    s=accuracy_score(y,self%predict(x))
  end function
end module
