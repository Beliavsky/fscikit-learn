module sklearn_neighbors
  use sklearn_nca, only: neighborhood_components_analysis
  use sklearn_neighbor_graph, only: kneighbors_transformer,radius_neighbors_transformer
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common
  use sklearn_outlier, only: local_outlier_factor
  use sklearn_density, only: kernel_density
  use sklearn_metrics, only: pairwise_distances,accuracy_score,r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: neighborhood_components_analysis
  public :: nearest_neighbors,kneighbors_classifier,kneighbors_regressor
  public :: radius_neighbors_classifier,radius_neighbors_regressor,nearest_centroid
  public :: local_outlier_factor
  public :: kernel_density
  public :: kneighbors_transformer,radius_neighbors_transformer,csr_matrix,csr_from_dense,csr_from_arrays
  type :: nearest_neighbors
    integer :: n_neighbors=5
    character(16) :: metric='minkowski'
    real(dp) :: p=2
    real(dp), allocatable :: x_(:,:)
  contains
    procedure :: fit=>neighbors_fit
    procedure :: kneighbors=>neighbors_query
    procedure :: radius_neighbors=>radius_query
    procedure :: kneighbors_graph=>neighbors_graph
  end type
  type :: kneighbors_classifier
    integer :: n_neighbors=5
    character(16) :: weights='uniform',metric='minkowski'
    real(dp) :: p=2
    type(nearest_neighbors) :: search_
    integer, allocatable :: y_(:),classes_(:)
  contains
    procedure :: fit=>knc_fit
    procedure :: predict_proba=>knc_proba
    procedure :: predict=>knc_predict
    procedure :: score=>knc_score
  end type
  type :: kneighbors_regressor
    integer :: n_neighbors=5
    character(16) :: weights='uniform',metric='minkowski'
    real(dp) :: p=2
    type(nearest_neighbors) :: search_
    real(dp), allocatable :: y_(:)
  contains
    procedure :: fit=>knr_fit
    procedure :: predict=>knr_predict
    procedure :: score=>knr_score
  end type
  type :: radius_neighbors_classifier
    real(dp) :: radius=1,p=2
    character(16) :: weights='uniform',metric='minkowski'
    logical :: allow_outliers=.false.
    integer :: outlier_label=-1
    integer, allocatable :: classes_(:),y_(:)
    real(dp), allocatable :: x_(:,:)
  contains
    procedure :: fit=>radius_classifier_fit
    procedure :: predict_proba=>radius_classifier_proba
    procedure :: predict=>radius_classifier_predict
    procedure :: score=>radius_classifier_score
  end type
  type :: radius_neighbors_regressor
    real(dp) :: radius=1,p=2
    character(16) :: weights='uniform',metric='minkowski'
    real(dp), allocatable :: x_(:,:),y_(:)
  contains
    procedure :: fit=>radius_regressor_fit
    procedure :: predict=>radius_regressor_predict
    procedure :: score=>radius_regressor_score
  end type
  type :: nearest_centroid
    character(16) :: metric='euclidean',prior_mode='uniform'
    real(dp) :: shrink_threshold=0
    integer :: n_features_in_=0
    real(dp), allocatable :: priors(:),class_prior_(:)
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: centroids_(:,:),within_class_std_dev_(:),deviations_(:,:)
  contains
    procedure :: fit=>centroid_fit
    procedure :: decision_function=>centroid_decision
    procedure :: predict_log_proba=>centroid_log_proba
    procedure :: predict_proba=>centroid_proba
    procedure :: predict=>centroid_predict
    procedure :: score=>centroid_score
  end type
contains
  subroutine neighbors_fit(self,x)
    class(nearest_neighbors), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x); call require(self%n_neighbors>0,'n_neighbors must be positive')
    self%x_=x
  end subroutine
  subroutine neighbors_query(self,x,distances,indices,n_neighbors)
    class(nearest_neighbors), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: distances(:,:)
    integer, allocatable, intent(out) :: indices(:,:)
    integer, intent(in), optional :: n_neighbors
    real(dp), allocatable :: d(:,:)
    integer, allocatable :: order(:)
    integer :: i,k
    call require(allocated(self%x_),'nearest_neighbors is not fitted')
    k=self%n_neighbors; if(present(n_neighbors)) k=n_neighbors
    call require(k>0.and.k<=size(self%x_,1),'n_neighbors exceeds training size')
    d=pairwise_distances(x,self%x_,self%metric,self%p)
    allocate(distances(size(x,1),k),indices(size(x,1),k))
    do i=1,size(x,1)
      order=argsort(d(i,:)); indices(i,:)=order(:k); distances(i,:)=d(i,order(:k))
    end do
  end subroutine
  function radius_query(self,x,radius) result(mask)
    class(nearest_neighbors), intent(in) :: self
    real(dp), intent(in) :: x(:,:),radius
    logical, allocatable :: mask(:,:)
    call require(allocated(self%x_),'nearest_neighbors is not fitted')
    call require(radius>=0,'radius must be nonnegative')
    mask=pairwise_distances(x,self%x_,self%metric,self%p)<=radius
  end function
  function neighbors_graph(self,x,mode) result(graph)
    class(nearest_neighbors), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in), optional :: mode
    real(dp), allocatable :: graph(:,:),d(:,:)
    integer, allocatable :: idx(:,:)
    integer :: i
    logical :: distance
    distance=.false.
    if(present(mode)) then
      call require(mode=='connectivity'.or.mode=='distance','unknown graph mode')
      distance=mode=='distance'
    end if
    call self%kneighbors(x,d,idx); allocate(graph(size(x,1),size(self%x_,1))); graph=0
    do i=1,size(x,1)
      if(distance) then
        graph(i,idx(i,:))=d(i,:)
      else
        graph(i,idx(i,:))=1
      end if
    end do
  end function
  subroutine knc_fit(self,x,y)
    class(kneighbors_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call check_xy(x,size(y)); self%y_=y; self%classes_=unique_labels(y)
    self%search_%n_neighbors=self%n_neighbors; self%search_%metric=self%metric; self%search_%p=self%p
    call self%search_%fit(x)
  end subroutine
  function get_weights(d,kind) result(w)
    real(dp), intent(in) :: d(:)
    character(*), intent(in) :: kind
    real(dp) :: w(size(d))
    select case(trim(kind))
    case('uniform'); w=1
    case('distance')
      if(any(d==0)) then
        w=merge(1._dp,0._dp,d==0)
      else
        w=1/d
      end if
    case default; call require(.false.,'unknown neighbor weights')
    end select
  end function
  function knc_proba(self,x) result(prob)
    class(kneighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),d(:,:),w(:)
    integer, allocatable :: idx(:,:)
    integer :: i,j,c
    call require(allocated(self%y_),'kneighbors_classifier is not fitted')
    call self%search_%kneighbors(x,d,idx)
    allocate(prob(size(x,1),size(self%classes_))); prob=0
    do i=1,size(x,1)
      w=get_weights(d(i,:),self%weights)
      do j=1,size(idx,2)
        c=label_index(self%classes_,self%y_(idx(i,j))); prob(i,c)=prob(i,c)+w(j)
      end do
      prob(i,:)=prob(i,:)/sum(w)
    end do
  end function
  function knc_predict(self,x) result(y)
    class(kneighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function knc_score(self,x,y) result(v)
    class(kneighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine knr_fit(self,x,y)
    class(kneighbors_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); self%y_=y
    self%search_%n_neighbors=self%n_neighbors; self%search_%metric=self%metric; self%search_%p=self%p
    call self%search_%fit(x)
  end subroutine
  function knr_predict(self,x) result(y)
    class(kneighbors_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    real(dp), allocatable :: d(:,:),w(:)
    integer, allocatable :: idx(:,:)
    integer :: i
    call require(allocated(self%y_),'kneighbors_regressor is not fitted')
    call self%search_%kneighbors(x,d,idx)
    do i=1,size(y)
      w=get_weights(d(i,:),self%weights); y(i)=dot_product(w,self%y_(idx(i,:)))/sum(w)
    end do
  end function
  real(dp) function knr_score(self,x,y) result(v)
    class(kneighbors_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
  subroutine radius_classifier_fit(self,x,y)
    class(radius_neighbors_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call check_xy(x,size(y)); call require(self%radius>0,'radius must be positive')
    call require(self%weights=='uniform'.or.self%weights=='distance','unknown neighbor weights')
    self%x_=x; self%y_=y; self%classes_=unique_labels(y)
  end subroutine
  function radius_classifier_proba(self,x) result(prob)
    class(radius_neighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),d(:,:),weights(:)
    integer, allocatable :: ids(:)
    integer :: i,j,c,n
    call require(allocated(self%x_),'radius classifier is not fitted')
    d=pairwise_distances(x,self%x_,self%metric,self%p); n=size(self%y_)
    allocate(prob(size(x,1),size(self%classes_))); prob=0
    do i=1,size(x,1)
      ids=pack([(j,j=1,n)],d(i,:)<=self%radius)
      if(size(ids)==0) then
        call require(self%allow_outliers,'query has no neighbors within radius')
        c=label_index(self%classes_,self%outlier_label)
        if(c>0) prob(i,c)=1
        cycle
      end if
      weights=get_weights(d(i,ids),self%weights)
      do j=1,size(ids)
        c=label_index(self%classes_,self%y_(ids(j))); prob(i,c)=prob(i,c)+weights(j)
      end do
      prob(i,:)=prob(i,:)/sum(weights)
    end do
  end function
  function radius_classifier_predict(self,x) result(y)
    class(radius_neighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      if(sum(prob(i,:))==0) then
        y(i)=self%outlier_label
      else
        y(i)=self%classes_(maxloc(prob(i,:),dim=1))
      end if
    end do
  end function
  real(dp) function radius_classifier_score(self,x,y) result(v)
    class(radius_neighbors_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine radius_regressor_fit(self,x,y)
    class(radius_neighbors_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%radius>0,'radius must be positive')
    call require(self%weights=='uniform'.or.self%weights=='distance','unknown neighbor weights')
    self%x_=x; self%y_=y
  end subroutine
  function radius_regressor_predict(self,x) result(y)
    use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
    class(radius_neighbors_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    real(dp), allocatable :: d(:,:),weights(:)
    integer, allocatable :: ids(:)
    integer :: i,j,n
    call require(allocated(self%x_),'radius regressor is not fitted')
    d=pairwise_distances(x,self%x_,self%metric,self%p); n=size(self%y_)
    do i=1,size(x,1)
      ids=pack([(j,j=1,n)],d(i,:)<=self%radius)
      if(size(ids)==0) then
        y(i)=ieee_value(0._dp,ieee_quiet_nan)
      else
        weights=get_weights(d(i,ids),self%weights)
        y(i)=dot_product(weights,self%y_(ids))/sum(weights)
      end if
    end do
  end function
  real(dp) function radius_regressor_score(self,x,y) result(v)
    class(radius_neighbors_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
  ! NearestCentroid 1.6.1: sklearn/neighbors/_nearest_centroid.py.
  ! BSD attribution retained in THIRD_PARTY_NOTICES.md.
  elemental real(dp) function centroid_ratio(numerator,denominator) result(ratio)
    real(dp), intent(in) :: numerator,denominator
    if(ieee_is_nan(denominator)) then
      ratio=ieee_value(0._dp,ieee_quiet_nan)
    else if(denominator==0) then
      if(numerator==0) then
        ratio=ieee_value(0._dp,ieee_quiet_nan)
      else
        ratio=sign(ieee_value(0._dp,ieee_positive_inf),numerator)
      end if
    else if(denominator<1.and.abs(numerator)/huge(1._dp)>denominator) then
      ratio=sign(ieee_value(0._dp,ieee_positive_inf),numerator)
    else
      ratio=numerator/denominator
    end if
  end function
  subroutine centroid_fit(self,x,y)
    class(nearest_centroid), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: mean(size(x,2)),s(size(x,2)),m,total,denominator,limit
    real(dp), allocatable :: values(:)
    integer, allocatable :: order(:),ids(:)
    integer :: c,k,n,j,nc,i,p
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_); n=size(y); p=size(x,2)
    call require(k>1,'nearest centroid requires multiple classes')
    call require(self%metric=='euclidean'.or.self%metric=='manhattan','unknown centroid metric')
    call require(ieee_is_finite(self%shrink_threshold),'centroid shrink_threshold must be finite')
    call require(self%shrink_threshold>=0,'centroid shrink_threshold must be nonnegative')
    call require(self%prior_mode=='uniform'.or.self%prior_mode=='empirical','unknown centroid prior_mode')
    limit=sqrt(huge(1._dp))/(16*sqrt(real(n,dp)*real(p,dp)))
    call require(maxval(abs(x))<=limit,'centroid data exceeds working range')
    call require(any(maxval(x,dim=1)>minval(x,dim=1)),'nearest centroid all features have zero variance')
    self%n_features_in_=p
    if(allocated(self%priors)) then
      call require(size(self%priors)==k,'centroid priors must match class count')
      call require(all(ieee_is_finite(self%priors)),'centroid priors must be finite')
      call require(all(self%priors>=0),'centroid priors must be nonnegative')
      call require(maxval(self%priors)<=huge(1._dp)/k,'centroid prior total exceeds numeric range')
      self%class_prior_=self%priors; total=sum(self%class_prior_)
      if(abs(total-1)>1.e-8_dp+1.e-5_dp) then
        if(total==0) then
          self%class_prior_=ieee_value(0._dp,ieee_quiet_nan)
        else
          self%class_prior_=self%class_prior_/total
        end if
      end if
    else if(self%prior_mode=='empirical') then
      self%class_prior_=real([(count(y==self%classes_(c)),c=1,k)],dp)/n
    else
      self%class_prior_=spread(1._dp/k,1,k)
    end if
    if(allocated(self%centroids_)) deallocate(self%centroids_,self%deviations_)
    allocate(self%centroids_(k,p),self%deviations_(k,p))
    self%within_class_std_dev_=spread(0._dp,1,p); mean=sum(x,dim=1)/n
    do c=1,k
      ids=pack([(i,i=1,n)],y==self%classes_(c)); nc=size(ids)
      do j=1,p
        values=x(ids,j)
        if(self%metric=='manhattan') then
          order=argsort(values)
          self%centroids_(c,j)=(values(order((nc+1)/2))+values(order(nc/2+1)))/2
        else
          self%centroids_(c,j)=sum(values)/nc
        end if
        self%within_class_std_dev_(j)=self%within_class_std_dev_(j)+sum((values-self%centroids_(c,j))**2)
      end do
    end do
    if(n==k) then
      self%within_class_std_dev_=ieee_value(0._dp,ieee_quiet_nan)
    else
      self%within_class_std_dev_=sqrt(self%within_class_std_dev_/(n-k))
    end if
    s=self%within_class_std_dev_
    if(.not.any(ieee_is_nan(s))) then
      order=argsort(s); j=p
      s=s+(s(order((j+1)/2))+s(order(j/2+1)))/2
    end if
    do c=1,k
      m=sqrt(1._dp/count(y==self%classes_(c))-1._dp/n)
      do j=1,p
        denominator=m*s(j)
        self%deviations_(c,j)=centroid_ratio(self%centroids_(c,j)-mean(j),denominator)
        if(self%shrink_threshold==0) cycle
        if(ieee_is_nan(self%deviations_(c,j))) then
          self%centroids_(c,j)=ieee_value(0._dp,ieee_quiet_nan)
        else
          self%deviations_(c,j)=sign(max(abs(self%deviations_(c,j))-self%shrink_threshold,0._dp), &
            self%deviations_(c,j))
          if(.not.ieee_is_finite(self%deviations_(c,j)).and.denominator==0) then
            self%centroids_(c,j)=ieee_value(0._dp,ieee_quiet_nan)
          else
            self%centroids_(c,j)=mean(j)+denominator*self%deviations_(c,j)
          end if
        end if
      end do
    end do
  end subroutine
  subroutine centroid_query_check(self,x)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%centroids_),'nearest centroid is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'centroid query feature count mismatch')
    call require(all(ieee_is_finite(self%centroids_)),'centroid query requires finite fitted centroids')
    call require(maxval(abs(x))<=sqrt(huge(1._dp))/(16*real(size(x,2),dp)), &
      'centroid query exceeds working range')
  end subroutine
  function centroid_discriminants(self,x) result(scores)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:),normalized(:,:),centers(:,:)
    real(dp) :: limit,scale
    integer :: j,c
    call centroid_query_check(self,x)
    call require(all(ieee_is_finite(self%within_class_std_dev_)), &
      'centroid discriminants require finite within-class deviations')
    normalized=x; centers=self%centroids_
    limit=sqrt(huge(1._dp))/(16*real(size(x,2),dp))
    do j=1,size(x,2)
      scale=self%within_class_std_dev_(j)
      if(scale==0) cycle
      if(scale<1) then
        call require(maxval(abs(normalized(:,j)))<=limit*scale.and.maxval(abs(centers(:,j)))<=limit*scale, &
          'centroid standardized values exceed working range')
      end if
      normalized(:,j)=normalized(:,j)/scale;centers(:,j)=centers(:,j)/scale
    end do
    call require(maxval(abs(normalized))<=limit.and.maxval(abs(centers))<=limit, &
      'centroid standardized values exceed working range')
    scores=-pairwise_distances(normalized,centers,self%metric)**2
    do c=1,size(self%classes_)
      if(ieee_is_nan(self%class_prior_(c))) then
        scores(:,c)=ieee_value(0._dp,ieee_quiet_nan)
      else if(self%class_prior_(c)==0) then
        scores(:,c)=ieee_value(0._dp,ieee_negative_inf)
      else
        scores(:,c)=scores(:,c)+2*log(self%class_prior_(c))
      end if
    end do
  end function
  function centroid_decision(self,x) result(scores)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:),raw(:,:)
    call require(self%metric=='euclidean','centroid decision/probability methods require euclidean metric')
    raw=centroid_discriminants(self,x)
    if(size(self%classes_)==2) then
      scores=reshape(raw(:,2)-raw(:,1),[size(x,1),1])
    else
      call move_alloc(raw,scores)
    end if
  end function
  function centroid_log_proba(self,x) result(scores)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    integer :: i
    call require(self%metric=='euclidean','centroid decision/probability methods require euclidean metric')
    scores=centroid_discriminants(self,x)
    do i=1,size(x,1)
      if(any(ieee_is_nan(scores(i,:)))) then
        scores(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else
        scores(i,:)=scores(i,:)-maxval(scores(i,:))
        scores(i,:)=scores(i,:)-log(sum(exp(scores(i,:))))
      end if
    end do
  end function
  function centroid_proba(self,x) result(scores)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    scores=exp(self%predict_log_proba(x))
  end function
  function centroid_predict(self,x) result(y)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i,c,index
    real(dp), allocatable :: d(:,:)
    logical :: uniform
    call centroid_query_check(self,x)
    uniform=.false.
    if(all(ieee_is_finite(self%class_prior_))) &
      uniform=all(abs(self%class_prior_-1._dp/size(self%classes_))<=1.e-8_dp+1.e-5_dp/size(self%classes_))
    if(uniform) then
      d=pairwise_distances(x,self%centroids_,self%metric)
      y=self%classes_(minloc(d,dim=2))
    else
      d=centroid_discriminants(self,x)
      do i=1,size(y)
        index=0
        do c=1,size(self%classes_)
          if(ieee_is_nan(d(i,c))) then
            index=c;exit
          end if
        end do
        if(index==0) index=maxloc(d(i,:),dim=1)
        y(i)=self%classes_(index)
      end do
    end if
  end function
  real(dp) function centroid_score(self,x,y) result(v)
    class(nearest_centroid), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
end module
