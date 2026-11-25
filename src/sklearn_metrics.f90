module sklearn_metrics
  use sklearn_error_metrics,only:mean_squared_error,root_mean_squared_error, &
   mean_absolute_error,mean_absolute_percentage_error,mean_squared_error_raw,root_mean_squared_error_raw, &
   mean_absolute_error_raw,mean_absolute_percentage_error_raw
  use sklearn_variance_scores,only:r2_score,explained_variance_score,r2_score_raw,explained_variance_score_raw
  use sklearn_confusion_matrix,only:confusion_matrix
  use sklearn_categorical_scores,only:balanced_accuracy_score,matthews_corrcoef,cohen_kappa_score
  use sklearn_likelihood_ratios,only:class_likelihood_ratios
  use sklearn_subset_metrics,only:accuracy_score,zero_one_loss,hamming_loss
  use sklearn_probability_metrics, only: log_loss,d2_log_loss_score
  use sklearn_bicluster_metrics, only: consensus_score,bicluster_similarity
  use sklearn_classification_metrics, only: precision_score,recall_score,f1_score,jaccard_score, &
    fbeta_score,precision_recall_fscore_support,multilabel_confusion_matrix
  use sklearn_pairwise_measures, only: cosine_distances,nan_euclidean_distances,haversine_distances, &
    paired_euclidean_distances,paired_manhattan_distances,paired_cosine_distances,paired_distances, &
    laplacian_kernel,additive_chi2_kernel,chi2_kernel
  use sklearn_score_ranking, only: hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
  use sklearn_regression_metrics, only: mean_pinball_loss,mean_tweedie_deviance,d2_tweedie_score, &
    d2_pinball_score,d2_absolute_error_score,root_mean_squared_log_error
  use sklearn_multilabel_ranking, only: coverage_error,label_ranking_average_precision_score,label_ranking_loss
  use sklearn_cluster_information, only: normalized_mutual_info_score,adjusted_mutual_info_score,fowlkes_mallows_score, &
    contingency_matrix,pair_confusion_matrix,entropy,expected_mutual_information,homogeneity_completeness_v_measure
  use sklearn_mutual_info, only: mutual_info_score
  use sklearn_common
  use sklearn_ranking, only: roc_curve,precision_recall_curve,det_curve,auc,roc_auc_score, &
    average_precision_score,brier_score_loss
  implicit none
  private
  public::class_likelihood_ratios
  public::zero_one_loss
  public :: d2_log_loss_score
  public :: consensus_score,bicluster_similarity
  public :: normalized_mutual_info_score,adjusted_mutual_info_score,fowlkes_mallows_score,contingency_matrix
  public :: pair_confusion_matrix,entropy,expected_mutual_information,homogeneity_completeness_v_measure
  public :: fbeta_score,precision_recall_fscore_support,multilabel_confusion_matrix
  public :: cosine_distances,nan_euclidean_distances,haversine_distances,paired_euclidean_distances
  public :: paired_manhattan_distances,paired_cosine_distances,paired_distances
  public :: laplacian_kernel,additive_chi2_kernel,chi2_kernel
  public :: hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
  public :: mean_pinball_loss,mean_tweedie_deviance,d2_tweedie_score,d2_pinball_score,d2_absolute_error_score
  public :: root_mean_squared_log_error
  public :: coverage_error,label_ranking_average_precision_score,label_ranking_loss
  public :: mutual_info_score
  public :: mean_squared_error, root_mean_squared_error, mean_absolute_error
  public :: median_absolute_error, mean_absolute_percentage_error, r2_score, explained_variance_score
  public :: r2_score_raw,explained_variance_score_raw
  public :: mean_squared_error_raw,root_mean_squared_error_raw
  public :: mean_absolute_error_raw,mean_absolute_percentage_error_raw
  public :: max_error, mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance
  public :: accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
  public :: balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, log_loss
  public :: roc_auc_score, average_precision_score, hamming_loss, jaccard_score
  public :: roc_curve,precision_recall_curve,det_curve,auc,brier_score_loss
  public :: pairwise_distances, euclidean_distances, manhattan_distances, cosine_similarity
  public :: linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
  public :: silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
  public :: adjusted_rand_score, rand_score, homogeneity_score, completeness_score, v_measure_score
contains
  subroutine check_targets(y,p)
    real(dp), intent(in) :: y(:),p(:)
    call require(size(y)>0.and.size(y)==size(p),'invalid metric target sizes')
    call check_x(reshape(y,[size(y),1])); call check_x(reshape(p,[size(p),1]))
  end subroutine
  real(dp) function median_absolute_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    real(dp) :: errors(size(y))
    integer :: order(size(y)),n
    call check_targets(y,p); errors=abs(y-p); order=argsort(errors); n=size(y)
    v=(errors(order((n+1)/2))+errors(order(n/2+1)))/2
  end function
  real(dp) function max_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); v=maxval(abs(y-p))
  end function
  real(dp) function mean_squared_log_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); call require(all(y>=0).and.all(p>=0),'MSLE requires nonnegative targets')
    v=sum((log(1+y)-log(1+p))**2)/size(y)
  end function
  real(dp) function mean_poisson_deviance(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    integer :: i
    call check_targets(y,p); call require(all(y>=0).and.all(p>0),'invalid Poisson targets')
    v=0
    do i=1,size(y)
      v=v+p(i)-y(i)
      if(y(i)>0) v=v+y(i)*log(y(i)/p(i))
    end do
    v=2*v/size(y)
  end function
  real(dp) function mean_gamma_deviance(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); call require(all(y>0).and.all(p>0),'invalid Gamma targets')
    v=2*sum(log(p/y)+y/p-1)/size(y)
  end function
  function pairwise_distances(x,y,metric,p) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    character(*), intent(in), optional :: metric
    real(dp), intent(in), optional :: p
    real(dp) :: d(size(x,1),size(y,1)),power,nx,ny
    integer :: i,j
    character(16) :: kind
    call check_x(x); call check_x(y); call require(size(x,2)==size(y,2),'pairwise feature mismatch')
    kind='euclidean'; if(present(metric)) kind=metric
    power=2; if(present(p)) power=p
    call require(power>=1,'Minkowski p must be >=1')
    do j=1,size(y,1)
      do i=1,size(x,1)
        select case(trim(kind))
        case('euclidean'); d(i,j)=sqrt(sum((x(i,:)-y(j,:))**2))
        case('sqeuclidean'); d(i,j)=sum((x(i,:)-y(j,:))**2)
        case('manhattan','cityblock'); d(i,j)=sum(abs(x(i,:)-y(j,:)))
        case('chebyshev'); d(i,j)=maxval(abs(x(i,:)-y(j,:)))
        case('minkowski'); d(i,j)=sum(abs(x(i,:)-y(j,:))**power)**(1/power)
        case('cosine')
          nx=sqrt(sum(x(i,:)**2)); ny=sqrt(sum(y(j,:)**2)); d(i,j)=1
          if(nx>0.and.ny>0) d(i,j)=max(0._dp,min(2._dp,1-dot_product(x(i,:),y(j,:))/(nx*ny)))
        case default; call require(.false.,'unknown distance metric')
        end select
      end do
    end do
  end function
  function euclidean_distances(x,y,squared) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    logical, intent(in), optional :: squared
    real(dp) :: d(size(x,1),size(y,1))
    d=pairwise_distances(x,y,'sqeuclidean')
    if(present(squared)) then
      if(squared) return
    end if
    d=sqrt(d)
  end function
  function manhattan_distances(x,y) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp) :: d(size(x,1),size(y,1))
    d=pairwise_distances(x,y,'manhattan')
  end function
  function cosine_similarity(x,y) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp) :: d(size(x,1),size(y,1))
    d=1-pairwise_distances(x,y,'cosine')
  end function
  function linear_kernel(x,y) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp) :: d(size(x,1),size(y,1))
    call check_x(x); call check_x(y); call require(size(x,2)==size(y,2),'kernel feature mismatch')
    d=matmul(x,transpose(y))
  end function
  function rbf_kernel(x,y,gamma) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), intent(in), optional :: gamma
    real(dp) :: d(size(x,1),size(y,1)),g
    g=1._dp/size(x,2); if(present(gamma)) g=gamma
    call require(g>=0,'gamma must be nonnegative')
    d=exp(-g*pairwise_distances(x,y,'sqeuclidean'))
  end function
  function polynomial_kernel(x,y,degree,gamma,coef0) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    integer, intent(in), optional :: degree
    real(dp), intent(in), optional :: gamma,coef0
    real(dp) :: d(size(x,1),size(y,1)),g,c
    integer :: power
    power=3; g=1._dp/size(x,2); c=1
    if(present(degree)) power=degree
    if(present(gamma)) g=gamma
    if(present(coef0)) c=coef0
    d=(g*linear_kernel(x,y)+c)**power
  end function
  function sigmoid_kernel(x,y,gamma,coef0) result(d)
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), intent(in), optional :: gamma,coef0
    real(dp) :: d(size(x,1),size(y,1)),g,c
    g=1._dp/size(x,2); c=1
    if(present(gamma)) g=gamma
    if(present(coef0)) c=coef0
    d=tanh(g*linear_kernel(x,y)+c)
  end function
  function silhouette_samples(x,labels) result(s)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: labels(:)
    real(dp) :: s(size(labels)),a,b,t
    real(dp), allocatable :: d(:,:)
    integer, allocatable :: classes(:)
    integer :: i,j,n
    call check_xy(x,size(labels)); classes=unique_labels(labels)
    call require(size(classes)>1.and.size(classes)<size(labels),'invalid number of clusters')
    d=pairwise_distances(x,x)
    do i=1,size(labels)
      n=count(labels==labels(i)); s(i)=0
      if(n==1) cycle
      a=sum(d(i,:),mask=labels==labels(i))/(n-1); b=huge(1._dp)
      do j=1,size(classes)
        if(classes(j)==labels(i)) cycle
        t=sum(d(i,:),mask=labels==classes(j))/count(labels==classes(j)); b=min(b,t)
      end do
      if(max(a,b)>0) s(i)=(b-a)/max(a,b)
    end do
  end function
  real(dp) function silhouette_score(x,labels) result(v)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: labels(:)
    v=sum(silhouette_samples(x,labels))/size(labels)
  end function
  real(dp) function calinski_harabasz_score(x,labels) result(v)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: labels(:)
    integer, allocatable :: classes(:)
    real(dp) :: mean(size(x,2)),center(size(x,2)),within,between
    integer :: i,j,n,k
    call check_xy(x,size(labels)); classes=unique_labels(labels); k=size(classes); n=size(labels)
    call require(k>1.and.k<n,'invalid number of clusters')
    mean=sum(x,dim=1)/n; within=0; between=0
    do i=1,k
      center=0
      do j=1,n
        if(labels(j)==classes(i)) center=center+x(j,:)
      end do
      center=center/count(labels==classes(i))
      between=between+count(labels==classes(i))*sum((center-mean)**2)
      do j=1,n
        if(labels(j)==classes(i)) within=within+sum((x(j,:)-center)**2)
      end do
    end do
    v=1; if(within>0) v=between*(n-k)/(within*(k-1))
  end function
  real(dp) function davies_bouldin_score(x,labels) result(v)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: labels(:)
    integer, allocatable :: classes(:)
    real(dp), allocatable :: centers(:,:),scatter(:),dist(:,:)
    real(dp) :: worst
    integer :: i,j,k,n
    call check_xy(x,size(labels)); classes=unique_labels(labels); k=size(classes); n=size(labels)
    call require(k>1.and.k<n,'invalid number of clusters')
    allocate(centers(k,size(x,2)),scatter(k)); centers=0; scatter=0
    do i=1,k
      do j=1,n
        if(labels(j)==classes(i)) centers(i,:)=centers(i,:)+x(j,:)
      end do
      centers(i,:)=centers(i,:)/count(labels==classes(i))
      do j=1,n
        if(labels(j)==classes(i)) scatter(i)=scatter(i)+sqrt(sum((x(j,:)-centers(i,:))**2))
      end do
      scatter(i)=scatter(i)/count(labels==classes(i))
    end do
    dist=pairwise_distances(centers,centers); v=0
    do i=1,k
      worst=0
      do j=1,k
        if(i==j.or.dist(i,j)==0) cycle
        worst=max(worst,(scatter(i)+scatter(j))/dist(i,j))
      end do
      v=v+worst
    end do
    v=v/k
  end function
  subroutine contingency(y,p,table)
    integer, intent(in) :: y(:),p(:)
    real(dp), allocatable, intent(out) :: table(:,:)
    integer, allocatable :: a(:),b(:)
    integer :: i
    call require(size(y)>0.and.size(y)==size(p),'invalid cluster labels')
    a=unique_labels(y); b=unique_labels(p); allocate(table(size(a),size(b))); table=0
    do i=1,size(y)
      table(label_index(a,y(i)),label_index(b,p(i)))=table(label_index(a,y(i)),label_index(b,p(i)))+1
    end do
  end subroutine
  real(dp) function adjusted_rand_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp), allocatable :: t(:,:),a(:),b(:)
    real(dp) :: pairs,sa,sb,st,expected,den
    call contingency(y,p,t); a=sum(t,dim=2); b=sum(t,dim=1)
    pairs=real(size(y),dp)*(size(y)-1)/2
    v=1; if(pairs==0) return
    sa=sum(a*(a-1)/2); sb=sum(b*(b-1)/2); st=sum(t*(t-1)/2)
    expected=sa*sb/pairs; den=(sa+sb)/2-expected
    if(den/=0) v=(st-expected)/den
  end function
  real(dp) function rand_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp), allocatable :: t(:,:),a(:),b(:)
    real(dp) :: pairs
    call contingency(y,p,t); a=sum(t,dim=2); b=sum(t,dim=1)
    pairs=real(size(y),dp)*(size(y)-1)/2; v=1
    if(pairs>0) v=1-(sum(a*(a-1)/2)+sum(b*(b-1)/2)-sum(t*(t-1)))/pairs
  end function
  real(dp) function homogeneity_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp) :: c,measure
    call homogeneity_completeness_v_measure(y,p,v,c,measure)
  end function
  real(dp) function completeness_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp) :: h,measure
    call homogeneity_completeness_v_measure(y,p,h,v,measure)
  end function
  real(dp) function v_measure_score(y,p,beta) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp), intent(in), optional :: beta
    real(dp) :: h,c
    call homogeneity_completeness_v_measure(y,p,h,c,v,beta)
  end function
end module
