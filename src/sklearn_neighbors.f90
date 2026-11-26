module sklearn_neighbors
  use,intrinsic::ieee_exceptions
  use sklearn_pairwise_reductions,only:row_distance
  use sklearn_pairwise_chunks,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
  use sklearn_nca, only: neighborhood_components_analysis
  use sklearn_label_binarization,only:label_set
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
  public :: neighbor_probability,label_set
  type :: neighbor_probability
    integer,allocatable::classes(:)
    real(dp),allocatable::values(:,:)
  end type
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
    integer,allocatable::y_matrix_(:,:)
    type(label_set),allocatable::classes_multi_(:)
    real(dp),private,allocatable::encoded_y_(:,:)
    logical::outputs_2d_=.false.
    integer::n_features_in_=0,n_outputs_=0,n_samples_fit_=0
  contains
    procedure :: fit_dense=>knc_fit,fit_matrix=>knc_fit_matrix
    generic::fit=>fit_dense,fit_matrix
    procedure :: predict_proba=>knc_proba
    procedure :: predict_proba_multioutput=>knc_proba_multioutput
    procedure :: predict=>knc_predict,predict_matrix=>knc_predict_matrix
    procedure :: score_dense=>knc_score,score_matrix=>knc_score_matrix
    generic::score=>score_dense,score_matrix
  end type
  type :: kneighbors_regressor
    integer :: n_neighbors=5
    character(16) :: weights='uniform',metric='minkowski'
    real(dp) :: p=2
    type(nearest_neighbors) :: search_
    real(dp), allocatable :: y_(:),y_matrix_(:,:)
    logical::matrix_target_=.false.
    integer::n_features_in_=0,n_outputs_=0,n_samples_fit_=0
  contains
    procedure :: fit_dense=>knr_fit,fit_matrix=>knr_fit_matrix
    generic::fit=>fit_dense,fit_matrix
    procedure :: predict=>knr_predict,predict_matrix=>knr_predict_matrix
    procedure :: score_dense=>knr_score,score_matrix=>knr_score_matrix
    generic::score=>score_dense,score_matrix
  end type
  type :: radius_neighbors_classifier
    real(dp) :: radius=1,p=2
    character(16) :: weights='uniform',metric='minkowski'
    logical :: allow_outliers=.false.
    integer :: outlier_label=-1
    integer, allocatable :: classes_(:),y_(:)
    real(dp), allocatable :: x_(:,:)
    character(20)::outlier_mode='legacy'
    integer,allocatable::outlier_labels(:),outlier_label_(:),y_matrix_(:,:)
    type(label_set),allocatable::classes_multi_(:)
    real(dp),private,allocatable::encoded_y_(:,:)
    character(20),private::outlier_mode_='legacy'
    character(16),private::metric_='minkowski'
    real(dp),private::p_=2
    logical::outputs_2d_=.false.
    integer::n_features_in_=0,n_outputs_=0,n_samples_fit_=0
  contains
    procedure :: fit_dense=>radius_classifier_fit,fit_matrix=>radius_classifier_fit_matrix
    generic::fit=>fit_dense,fit_matrix
    procedure :: predict_proba=>radius_classifier_proba
    procedure :: predict_proba_multioutput=>radius_classifier_proba_multioutput
    procedure :: predict=>radius_classifier_predict,predict_matrix=>radius_classifier_predict_matrix
    procedure :: score_dense=>radius_classifier_score,score_matrix=>radius_classifier_score_matrix
    generic::score=>score_dense,score_matrix
  end type
  type :: radius_neighbors_regressor
    real(dp) :: radius=1,p=2
    character(16) :: weights='uniform',metric='minkowski'
    real(dp), allocatable :: x_(:,:),y_(:),y_matrix_(:,:)
    logical::matrix_target_=.false.
    integer::n_features_in_=0,n_outputs_=0,n_samples_fit_=0
    character(16),private::metric_='minkowski'
    real(dp),private::p_=2
  contains
    procedure :: fit_dense=>radius_regressor_fit,fit_matrix=>radius_regressor_fit_matrix
    generic::fit=>fit_dense,fit_matrix
    procedure :: predict=>radius_regressor_predict,predict_matrix=>radius_regressor_predict_matrix
    procedure :: score_dense=>radius_regressor_score,score_matrix=>radius_regressor_score_matrix
    generic::score=>score_dense,score_matrix
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
    class(kneighbors_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    call knc_fit_matrix(self,x,reshape(y,[size(y),1]))
  end subroutine
  subroutine knc_fit_matrix(self,x,y)
    class(kneighbors_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    integer::i,j
    call check_xy(x,size(y,1));call require(size(y,2)>0,'neighbor classification needs target columns')
    call regression_parameters(x,self%metric,self%p,self%weights)
    self%search_%n_neighbors=self%n_neighbors; self%search_%metric=self%metric; self%search_%p=self%p
    call self%search_%fit(x)
    self%y_matrix_=y;self%outputs_2d_=size(y,2)>1
    self%n_features_in_=size(x,2);self%n_outputs_=size(y,2);self%n_samples_fit_=size(x,1)
    if(allocated(self%classes_multi_))deallocate(self%classes_multi_)
    allocate(self%classes_multi_(size(y,2)))
    self%encoded_y_=real(y,dp)
    do j=1,size(y,2)
      self%classes_multi_(j)%values=unique_labels(y(:,j))
      do i=1,size(y,1)
        self%encoded_y_(i,j)=real(label_index(self%classes_multi_(j)%values,y(i,j)),dp)
      end do
    end do
    if(allocated(self%y_))deallocate(self%y_)
    if(allocated(self%classes_))deallocate(self%classes_)
    if(.not.self%outputs_2d_)then
      self%y_=y(:,1);self%classes_=self%classes_multi_(1)%values
    end if
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
    class(kneighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    real(dp),allocatable::prob(:,:)
    type(neighbor_probability),allocatable::all_prob(:)
    call require(.not.self%outputs_2d_,'joint neighbor probabilities require predict_proba_multioutput')
    all_prob=self%predict_proba_multioutput(x);prob=all_prob(1)%values
  end function
  function knc_proba_multioutput(self,x)result(prob)
    class(kneighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    type(neighbor_probability),allocatable::prob(:)
    real(dp),allocatable::flat(:,:)
    integer,allocatable::class_sizes(:)
    integer::j,first,last
    call require(allocated(self%y_matrix_),'kneighbors_classifier is not fitted')
    class_sizes=[(size(self%classes_multi_(j)%values),j=1,self%n_outputs_)]
    flat=regression_predict(self%search_%x_,self%encoded_y_,self%search_%metric,self%search_%p, &
      self%weights,self%n_neighbors,0._dp,.false.,x,class_sizes)
    allocate(prob(self%n_outputs_));first=1
    do j=1,self%n_outputs_
      last=first+class_sizes(j)-1
      prob(j)%classes=self%classes_multi_(j)%values;prob(j)%values=flat(:,first:last);first=last+1
    end do
  end function
  function knc_predict(self,x) result(y)
    class(kneighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    integer,allocatable::y(:),matrix(:,:)
    call require(.not.self%outputs_2d_,'joint neighbor labels require predict_matrix')
    matrix=self%predict_matrix(x);y=matrix(:,1)
  end function
  function knc_predict_matrix(self,x)result(y)
    class(kneighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    integer,allocatable::y(:,:),class_sizes(:)
    real(dp),allocatable::codes(:,:)
    integer::j
    call require(allocated(self%y_matrix_),'kneighbors_classifier is not fitted')
    class_sizes=[(size(self%classes_multi_(j)%values),j=1,self%n_outputs_)]
    codes=regression_predict(self%search_%x_,self%encoded_y_,self%search_%metric,self%search_%p, &
      self%weights,self%n_neighbors,0._dp,.false.,x,class_sizes,.true.)
    allocate(y(size(codes,1),self%n_outputs_))
    do j=1,self%n_outputs_
      y(:,j)=self%classes_multi_(j)%values(int(codes(:,j)))
    end do
  end function
  real(dp)function knc_score(self,x,y,sample_weight)result(v)
    class(kneighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    integer,intent(in)::y(:)
    v=accuracy_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  real(dp)function knc_score_matrix(self,x,y,sample_weight)result(v)
    class(kneighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    integer,intent(in)::y(:,:)
    v=accuracy_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  ! Regression prediction follows sklearn 1.6.1 neighbors/_regression.py and
  ! _base.py (BSD). Owned matrix targets and streaming adapters are native MIT.
  recursive real(dp)function neighbor_sum(a)result(value)
    real(dp),intent(in)::a(:)
    real(dp)::lane(8)
    integer::n,i,whole,half
    n=size(a);value=-0._dp
    if(n<8)then
      do i=1,n
        value=value+a(i)
      end do
    else if(n<=128)then
      lane=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        lane=lane+a(i:i+7)
      end do
      value=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
      do i=whole+1,n
        value=value+a(i)
      end do
    else
      half=n/2;half=half-mod(half,8);value=neighbor_sum(a(:half))+neighbor_sum(a(half+1:))
    end if
  end function
  subroutine regression_parameters(x,metric,p,weights)
    real(dp),intent(in)::x(:,:),p
    character(*),intent(in)::metric,weights
    call check_x(x)
    call require(weights=='uniform'.or.weights=='distance','unknown neighbor weights')
    call require(any(metric==[character(16)::'minkowski','euclidean','sqeuclidean','l2','manhattan', &
      'cityblock','l1','chebyshev','cosine','precomputed']),'unsupported neighbor regression metric')
    call require(.not.ieee_is_nan(p),'neighbor regression p is NaN')
    call require(p>0,'neighbor regression p must be positive')
    if(metric=='precomputed')then
      call require(size(x,1)==size(x,2),'precomputed neighbor fit must be square')
      call require(all(x>=0),'precomputed neighbor distances must be nonnegative')
    end if
  end subroutine
  function regression_average(targets,d,weights,knn)result(value)
    real(dp),intent(in)::targets(:,:),d(:)
    character(*),intent(in)::weights
    logical,intent(in)::knn
    real(dp)::value(size(targets,2)),w(size(d)),denom
    logical::infmask(size(d))
    integer::i,j
    if(weights=='uniform')then
      if(size(targets,2)==1)then
        value(1)=neighbor_sum(targets(:,1))/size(d)
      else
        value=0
        do i=1,size(d)
          value=value+targets(i,:)
        end do
        value=value/size(d)
      end if
      return
    end if
    call require(weights=='distance','unknown neighbor weights')
    if(knn)then
      w=1/d;infmask=.not.ieee_is_finite(w).and..not.ieee_is_nan(w)
      if(any(infmask))w=merge(1._dp,0._dp,infmask)
    else if(any(d==0))then
      w=merge(1._dp,0._dp,d==0)
    else
      w=1/d
    end if
    denom=neighbor_sum(w)
    if(knn.or.size(targets,2)==1)then
      do j=1,size(value)
        value(j)=neighbor_sum(w*targets(:,j))/denom
      end do
    else
      value=0
      do i=1,size(d)
        value=value+w(i)*targets(i,:)
      end do
      value=value/denom
    end if
  end function
  function classification_reduce(codes,d,weights,class_sizes,mode,by_radius)result(value)
    real(dp),intent(in)::codes(:,:),d(:)
    character(*),intent(in)::weights
    integer,intent(in)::class_sizes(:)
    logical,intent(in)::mode,by_radius
    real(dp),allocatable::value(:),votes(:)
    real(dp)::w(size(d)),normalizer
    logical::infmask(size(d))
    integer::i,j,c,first
    w=1
    if(weights=='distance')then
      if(by_radius)then
        if(any(d==0))then
          w=merge(1._dp,0._dp,d==0)
        else
          w=1/d
        end if
      else
        w=1/d;infmask=.not.ieee_is_finite(w).and..not.ieee_is_nan(w)
        if(any(infmask))w=merge(1._dp,0._dp,infmask)
      end if
    end if
    if(.not.by_radius)call require(any(w/=0),'all neighbors have zero weights')
    if(mode)then
      allocate(value(size(class_sizes)))
    else
      allocate(value(sum(class_sizes)))
    end if
    first=1
    do j=1,size(class_sizes)
      allocate(votes(class_sizes(j)));votes=0
      if(mode)then
        ! weighted_mode reduces a masked row per class, separately from the
        ! sequential additions used by predict_proba.
        do c=1,class_sizes(j)
          votes(c)=neighbor_sum(merge(w,0._dp,int(codes(:,j))==c))
        end do
        value(j)=real(maxloc(votes,dim=1),dp)
      else
        do i=1,size(d)
          c=int(codes(i,j));votes(c)=votes(c)+w(i)
        end do
        normalizer=neighbor_sum(votes)
        if(by_radius)then
          if(normalizer==0)normalizer=1
        end if
        value(first:first+class_sizes(j)-1)=votes/normalizer
        first=first+class_sizes(j)
      end if
      deallocate(votes)
    end do
  end function
  function regression_predict(training,targets,metric,p,weights,k,radius,by_radius,x, &
      class_sizes,vote_mode,empty_neighbors)result(prediction)
    real(dp),intent(in)::training(:,:),targets(:,:),p,radius
    character(*),intent(in)::metric,weights
    integer,intent(in)::k
    logical,intent(in)::by_radius
    real(dp),optional,intent(in)::x(:,:)
    integer,optional,intent(in)::class_sizes(:)
    logical,optional,intent(in)::vote_mode
    logical,allocatable,optional,intent(out)::empty_neighbors(:)
    real(dp),allocatable::prediction(:,:),dist(:,:)
    integer,allocatable::order(:),ids(:)
    logical,allocatable::keep(:)
    type(pairwise_distance_iterator)::state
    type(distance_chunk)::chunk
    character(16)::kind
    logical::self_query,available,modes(size(ieee_all)),flags(size(ieee_all)),mode
    integer::n,m,first,i,row,j,take,span,blockrow,trainrow,qrow,count_rows,outputs
    n=size(training,1);self_query=.not.present(x);m=n
    call require(weights=='uniform'.or.weights=='distance','unknown neighbor weights')
    if(present(x))then
      call check_x(x);m=size(x,1)
      call require(size(x,2)==size(training,2),'neighbor regression query feature count mismatch')
      if(metric=='precomputed')call require(all(x>=0),'precomputed neighbor distances must be nonnegative')
    end if
    if(.not.by_radius)then
      call require(k>0,'n_neighbors must be positive')
      call require(k<=n,'n_neighbors exceeds available training neighbors')
      if(self_query)call require(k<n,'training prediction needs fewer neighbors than training samples')
      take=k;if(self_query)take=k+1
    else
      call require(.not.ieee_is_nan(radius),'radius is NaN')
      call require(radius>=0,'radius must be nonnegative')
    end if
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    mode=.false.;if(present(vote_mode))mode=vote_mode
    outputs=size(targets,2)
    if(present(class_sizes))then
      if(.not.mode)outputs=sum(class_sizes)
    end if
    allocate(prediction(m,outputs))
    if(present(empty_neighbors))allocate(empty_neighbors(m))
    kind=metric
    if(kind=='minkowski')then
      if(p==1)kind='manhattan'
      if(p==2)kind='euclidean'
      if(.not.ieee_is_finite(p))kind='chebyshev'
    end if
    if(metric=='precomputed')then
      dist=training;if(present(x))dist=x
      first=0;call apply_block(dist,first)
    else if(kind=='euclidean'.or.kind=='l2'.or.kind=='sqeuclidean')then
      ! Neighbor reductions use ordered row norms, rather than the general
      ! pairwise GEMM path. Keep exact zero geometry for coincident rows.
      span=max(1,min(m,1048576/n))
      do first=0,m-1,span
        count_rows=min(span,m-first)
        if(allocated(dist))deallocate(dist)
        allocate(dist(count_rows,n))
        do trainrow=1,n
          do blockrow=1,count_rows
            qrow=first+blockrow
            if(present(x))then
              dist(blockrow,trainrow)=row_distance(x(qrow,:),training(trainrow,:),1,p,.false.,.false.,.false.)
            else
              dist(blockrow,trainrow)=row_distance(training(qrow,:),training(trainrow,:),1,p,.false.,.false.,.false.)
            end if
            if(kind/='sqeuclidean')dist(blockrow,trainrow)=sqrt(dist(blockrow,trainrow))
          end do
        end do
        call apply_block(dist,first)
      end do
    else
      if(present(x))then
        if(kind=='minkowski')then
          state=pairwise_distances_chunked(x,training,kind,working_memory=8._dp,p=p)
        else
          state=pairwise_distances_chunked(x,training,kind,working_memory=8._dp)
        end if
      else
        if(kind=='minkowski')then
          state=pairwise_distances_chunked(training,metric=kind,working_memory=8._dp,p=p)
        else
          state=pairwise_distances_chunked(training,metric=kind,working_memory=8._dp)
        end if
      end if
      do
        available=state%next(chunk);if(.not.available)exit
        call apply_block(chunk%distance%values,chunk%distance%start)
      end do
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  contains
    subroutine apply_block(d,start)
      real(dp),intent(in)::d(:,:)
      integer,intent(in)::start
      do i=1,size(d,1)
        row=start+i
        if(by_radius)then
          keep=d(i,:)<=radius
          if(self_query)keep(row)=.false.
          ids=pack([(j,j=1,n)],keep)
        else
          order=argsort(d(i,:));ids=order(:take)
          if(self_query)then
            keep=ids/=row
            if(all(keep))keep(1)=.false.
            ids=pack(ids,keep)
          end if
        end if
        if(present(empty_neighbors))empty_neighbors(row)=size(ids)==0
        if(size(ids)==0)then
          if(present(class_sizes).and.by_radius)then
            prediction(row,:)=0
          else
            prediction(row,:)=ieee_value(0._dp,ieee_quiet_nan)
          end if
        else if(present(class_sizes))then
          prediction(row,:)=classification_reduce(targets(ids,:),d(i,ids),weights,class_sizes,mode,by_radius)
        else
          prediction(row,:)=regression_average(targets(ids,:),d(i,ids),weights,.not.by_radius)
        end if
      end do
    end subroutine
  end function
  subroutine knr_fit(self,x,y)
    class(kneighbors_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    call knr_fit_common(self,x,reshape(y,[size(y),1]),.false.)
  end subroutine
  subroutine knr_fit_matrix(self,x,y)
    class(kneighbors_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    call knr_fit_common(self,x,y,.true.)
  end subroutine
  subroutine knr_fit_common(self,x,y,matrix_target)
    class(kneighbors_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    logical,intent(in)::matrix_target
    call check_xy(x,size(y,1));call check_x(y)
    call regression_parameters(x,self%metric,self%p,self%weights)
    self%search_%n_neighbors=self%n_neighbors;self%search_%metric=self%metric;self%search_%p=self%p
    call self%search_%fit(x)
    self%y_matrix_=y;self%matrix_target_=matrix_target
    self%n_outputs_=size(y,2);self%n_features_in_=size(x,2);self%n_samples_fit_=size(x,1)
    if(allocated(self%y_))deallocate(self%y_)
    if(.not.matrix_target)self%y_=y(:,1)
  end subroutine
  function knr_predict_matrix(self,x)result(y)
    class(kneighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    real(dp),allocatable::y(:,:)
    call require(allocated(self%y_matrix_),'kneighbors_regressor is not fitted')
    y=regression_predict(self%search_%x_,self%y_matrix_,self%search_%metric,self%search_%p, &
      self%weights,self%n_neighbors,0._dp,.false.,x)
  end function
  function knr_predict(self,x)result(y)
    class(kneighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    real(dp),allocatable::y(:),matrix(:,:)
    call require(.not.self%matrix_target_,'matrix-target neighbor regression requires predict_matrix')
    matrix=self%predict_matrix(x);y=matrix(:,1)
  end function
  real(dp)function knr_score(self,x,y,sample_weight)result(v)
    class(kneighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    real(dp),intent(in)::y(:)
    v=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  real(dp)function knr_score_matrix(self,x,y,sample_weight)result(v)
    class(kneighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    real(dp),intent(in)::y(:,:)
    v=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  subroutine radius_classifier_fit(self,x,y)
    class(radius_neighbors_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    call radius_classifier_fit_matrix(self,x,reshape(y,[size(y),1]))
  end subroutine
  subroutine radius_classifier_fit_matrix(self,x,y)
    class(radius_neighbors_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    integer::i,j,c
    integer,allocatable::counts(:)
    logical::enabled
    call check_xy(x,size(y,1));call require(size(y,2)>0,'radius classification needs target columns')
    call regression_parameters(x,self%metric,self%p,self%weights)
    call require(.not.ieee_is_nan(self%radius),'radius is NaN')
    call require(self%radius>=0,'radius must be nonnegative')
    call require(any(self%outlier_mode==[character(20)::'legacy','error','label','most_frequent']), &
      'unknown radius outlier mode')
    self%x_=x;self%y_matrix_=y;self%metric_=self%metric;self%p_=self%p
    self%outputs_2d_=size(y,2)>1;self%n_outputs_=size(y,2)
    self%n_features_in_=size(x,2);self%n_samples_fit_=size(x,1)
    self%outlier_mode_=self%outlier_mode
    if(allocated(self%classes_multi_))deallocate(self%classes_multi_)
    allocate(self%classes_multi_(self%n_outputs_));self%encoded_y_=real(y,dp)
    do j=1,self%n_outputs_
      self%classes_multi_(j)%values=unique_labels(y(:,j))
      do i=1,size(y,1)
        self%encoded_y_(i,j)=real(label_index(self%classes_multi_(j)%values,y(i,j)),dp)
      end do
    end do
    if(allocated(self%classes_))deallocate(self%classes_)
    if(allocated(self%y_))deallocate(self%y_)
    if(.not.self%outputs_2d_)then
      self%classes_=self%classes_multi_(1)%values;self%y_=y(:,1)
    end if
    if(allocated(self%outlier_label_))deallocate(self%outlier_label_)
    enabled=self%outlier_mode_=='label'.or.self%outlier_mode_=='most_frequent'
    if(self%outlier_mode_=='legacy')enabled=self%allow_outliers
    if(enabled)then
      allocate(self%outlier_label_(self%n_outputs_));self%outlier_label_=self%outlier_label
      if(self%outlier_mode_=='most_frequent')then
        do j=1,self%n_outputs_
          counts=[(count(y(:,j)==self%classes_multi_(j)%values(c)),c=1,size(self%classes_multi_(j)%values))]
          self%outlier_label_(j)=self%classes_multi_(j)%values(maxloc(counts,dim=1))
        end do
      else if(allocated(self%outlier_labels))then
        call require(size(self%outlier_labels)==self%n_outputs_,'outlier label count differs from target count')
        self%outlier_label_=self%outlier_labels
      end if
    end if
  end subroutine
  subroutine radius_outlier_labels(self,labels,enabled)
    class(radius_neighbors_classifier),intent(in)::self
    integer,allocatable,intent(out)::labels(:)
    logical,intent(out)::enabled
    allocate(labels(self%n_outputs_));labels=0
    if(self%outlier_mode_=='legacy')then
      enabled=self%allow_outliers
      if(enabled)then
        labels=self%outlier_label
        if(allocated(self%outlier_labels))then
          call require(size(self%outlier_labels)==self%n_outputs_,'outlier label count differs from target count')
          labels=self%outlier_labels
        end if
      end if
    else
      enabled=allocated(self%outlier_label_)
      if(enabled)labels=self%outlier_label_
    end if
  end subroutine
  function radius_classifier_proba(self,x) result(prob)
    class(radius_neighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    real(dp),allocatable::prob(:,:)
    type(neighbor_probability),allocatable::all_prob(:)
    call require(.not.self%outputs_2d_,'joint radius probabilities require predict_proba_multioutput')
    all_prob=self%predict_proba_multioutput(x);prob=all_prob(1)%values
  end function
  function radius_classifier_proba_multioutput(self,x)result(prob)
    class(radius_neighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    type(neighbor_probability),allocatable::prob(:)
    real(dp),allocatable::flat(:,:)
    integer,allocatable::class_sizes(:),labels(:)
    logical,allocatable::empty(:)
    logical::enabled
    integer::j,first,last,c,i
    call require(allocated(self%y_matrix_),'radius classifier is not fitted')
    class_sizes=[(size(self%classes_multi_(j)%values),j=1,self%n_outputs_)]
    flat=regression_predict(self%x_,self%encoded_y_,self%metric_,self%p_,self%weights,0,self%radius,.true.,x, &
      class_sizes=class_sizes,empty_neighbors=empty)
    call radius_outlier_labels(self,labels,enabled)
    if(any(empty))call require(enabled,'query has no neighbors within radius and no outlier label')
    allocate(prob(self%n_outputs_));first=1
    do j=1,self%n_outputs_
      last=first+class_sizes(j)-1
      prob(j)%classes=self%classes_multi_(j)%values;prob(j)%values=flat(:,first:last)
      if(enabled)then
        c=label_index(prob(j)%classes,labels(j))
        if(c>0)then
          do i=1,size(empty)
            if(empty(i))prob(j)%values(i,c)=1
          end do
        end if
      end if
      first=last+1
    end do
  end function
  function radius_classifier_predict(self,x) result(y)
    class(radius_neighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    integer,allocatable::y(:),matrix(:,:)
    call require(.not.self%outputs_2d_,'joint radius labels require predict_matrix')
    matrix=self%predict_matrix(x);y=matrix(:,1)
  end function
  function radius_classifier_predict_matrix(self,x)result(y)
    class(radius_neighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    integer,allocatable::y(:,:),labels(:)
    type(neighbor_probability),allocatable::prob(:)
    logical::enabled
    integer::i,j,c
    prob=self%predict_proba_multioutput(x);call radius_outlier_labels(self,labels,enabled)
    allocate(y(size(prob(1)%values,1),self%n_outputs_))
    do j=1,self%n_outputs_
      do i=1,size(y,1)
        c=findloc(ieee_is_nan(prob(j)%values(i,:)),.true.,dim=1)
        if(c>0)then
          y(i,j)=prob(j)%classes(c)
        else if(all(prob(j)%values(i,:)==0))then
          call require(enabled,'zero class probabilities require an outlier label')
          y(i,j)=labels(j)
        else
          y(i,j)=prob(j)%classes(maxloc(prob(j)%values(i,:),dim=1))
        end if
      end do
    end do
  end function
  real(dp)function radius_classifier_score(self,x,y,sample_weight)result(v)
    class(radius_neighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    integer,intent(in)::y(:)
    v=accuracy_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  real(dp)function radius_classifier_score_matrix(self,x,y,sample_weight)result(v)
    class(radius_neighbors_classifier),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    integer,intent(in)::y(:,:)
    v=accuracy_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  subroutine radius_regressor_fit(self,x,y)
    class(radius_neighbors_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    call radius_regressor_fit_common(self,x,reshape(y,[size(y),1]),.false.)
  end subroutine
  subroutine radius_regressor_fit_matrix(self,x,y)
    class(radius_neighbors_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    call radius_regressor_fit_common(self,x,y,.true.)
  end subroutine
  subroutine radius_regressor_fit_common(self,x,y,matrix_target)
    class(radius_neighbors_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    logical,intent(in)::matrix_target
    call check_xy(x,size(y,1));call check_x(y)
    call regression_parameters(x,self%metric,self%p,self%weights)
    call require(.not.ieee_is_nan(self%radius),'radius is NaN')
    call require(self%radius>=0,'radius must be nonnegative')
    self%x_=x;self%y_matrix_=y;self%matrix_target_=matrix_target
    self%metric_=self%metric;self%p_=self%p
    self%n_features_in_=size(x,2);self%n_outputs_=size(y,2);self%n_samples_fit_=size(x,1)
    if(allocated(self%y_))deallocate(self%y_)
    if(.not.matrix_target)self%y_=y(:,1)
  end subroutine
  function radius_regressor_predict_matrix(self,x)result(y)
    class(radius_neighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    real(dp),allocatable::y(:,:)
    call require(allocated(self%y_matrix_),'radius regressor is not fitted')
    y=regression_predict(self%x_,self%y_matrix_,self%metric_,self%p_,self%weights,0,self%radius,.true.,x)
  end function
  function radius_regressor_predict(self,x)result(y)
    class(radius_neighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:)
    real(dp),allocatable::y(:),matrix(:,:)
    call require(.not.self%matrix_target_,'matrix-target neighbor regression requires predict_matrix')
    matrix=self%predict_matrix(x);y=matrix(:,1)
  end function
  real(dp)function radius_regressor_score(self,x,y,sample_weight)result(v)
    class(radius_neighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    real(dp),intent(in)::y(:)
    v=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  real(dp)function radius_regressor_score_matrix(self,x,y,sample_weight)result(v)
    class(radius_neighbors_regressor),intent(in)::self
    real(dp),optional,intent(in)::x(:,:),sample_weight(:)
    real(dp),intent(in)::y(:,:)
    v=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
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
