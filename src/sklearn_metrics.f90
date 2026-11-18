module sklearn_metrics
  use sklearn_common
  implicit none
  private
  public :: mean_squared_error, root_mean_squared_error, mean_absolute_error
  public :: median_absolute_error, mean_absolute_percentage_error, r2_score, explained_variance_score
  public :: max_error, mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance
  public :: accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
  public :: balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, log_loss
  public :: roc_auc_score, average_precision_score, hamming_loss, jaccard_score
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
  real(dp) function mean_squared_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); v=sum((y-p)**2)/size(y)
  end function
  real(dp) function root_mean_squared_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    v=sqrt(mean_squared_error(y,p))
  end function
  real(dp) function mean_absolute_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); v=sum(abs(y-p))/size(y)
  end function
  real(dp) function median_absolute_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    real(dp) :: errors(size(y))
    integer :: order(size(y)),n
    call check_targets(y,p); errors=abs(y-p); order=argsort(errors); n=size(y)
    v=(errors(order((n+1)/2))+errors(order(n/2+1)))/2
  end function
  real(dp) function mean_absolute_percentage_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); v=sum(abs(y-p)/max(abs(y),epsilon(1._dp)))/size(y)
  end function
  real(dp) function max_error(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    call check_targets(y,p); v=maxval(abs(y-p))
  end function
  real(dp) function r2_score(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    real(dp) :: num,den
    call check_targets(y,p); call require(size(y)>1,'R2 requires at least two samples')
    num=sum((y-p)**2); den=sum((y-sum(y)/size(y))**2)
    if(den==0) then
      v=merge(1._dp,0._dp,num==0)
    else
      v=1-num/den
    end if
  end function
  real(dp) function explained_variance_score(y,p) result(v)
    real(dp), intent(in) :: y(:),p(:)
    real(dp) :: num,den
    call check_targets(y,p)
    num=sum((y-p-sum(y-p)/size(y))**2); den=sum((y-sum(y)/size(y))**2)
    if(den==0) then
      v=merge(1._dp,0._dp,num==0)
    else
      v=1-num/den
    end if
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
  real(dp) function accuracy_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    call require(size(y)>0.and.size(y)==size(p),'invalid label sizes')
    v=real(count(y==p),dp)/size(y)
  end function
  function confusion_matrix(y,p,labels) result(cm)
    integer, intent(in) :: y(:),p(:)
    integer, intent(in), optional :: labels(:)
    integer, allocatable :: cm(:,:), classes(:)
    integer :: i,a,b
    call require(size(y)>0.and.size(y)==size(p),'invalid label sizes')
    classes=unique_labels([y,p]); if(present(labels)) classes=labels
    allocate(cm(size(classes),size(classes))); cm=0
    do i=1,size(y)
      a=label_index(classes,y(i)); b=label_index(classes,p(i))
      if(a>0.and.b>0) cm(a,b)=cm(a,b)+1
    end do
  end function
  real(dp) function prf(y,p,kind,average,pos_label) result(v)
    integer, intent(in) :: y(:),p(:),kind
    character(*), intent(in), optional :: average
    integer, intent(in), optional :: pos_label
    integer, allocatable :: cm(:,:),classes(:)
    real(dp), allocatable :: tp(:),fp(:),fn(:),support(:),scores(:)
    real(dp) :: a,b,c
    integer :: pos,k
    character(16) :: avg
    avg='binary'; if(present(average)) avg=average
    pos=1; if(present(pos_label)) pos=pos_label
    classes=unique_labels([y,p]); cm=confusion_matrix(y,p); k=size(classes)
    allocate(tp(k)); tp=[(real(cm(pos,pos),dp),pos=1,k)]
    fp=real(sum(cm,dim=1),dp)-tp; fn=real(sum(cm,dim=2),dp)-tp; support=tp+fn
    pos=1; if(present(pos_label)) pos=pos_label
    if(trim(avg)=='micro') then
      a=sum(tp); b=sum(fp); c=sum(fn); v=score(a,b,c); return
    end if
    allocate(scores(k))
    do k=1,size(classes)
      scores(k)=score(tp(k),fp(k),fn(k))
    end do
    select case(trim(avg))
    case('binary')
      call require(size(classes)<=2,'binary average requires at most two classes')
      k=label_index(classes,pos); v=0
      if(k>0) v=scores(k)
    case('macro'); v=sum(scores)/size(scores)
    case('weighted'); v=dot_product(scores,support)/sum(support)
    case default; call require(.false.,'unknown metric average')
    end select
  contains
    real(dp) function score(tp,fp,fn) result(s)
      real(dp), intent(in) :: tp,fp,fn
      real(dp) :: den
      select case(kind)
      case(1); den=tp+fp; s=tp
      case(2); den=tp+fn; s=tp
      case(3); den=2*tp+fp+fn; s=2*tp
      case(4); den=tp+fp+fn; s=tp
      end select
      if(den>0) then
        s=s/den
      else
        s=0
      end if
    end function
  end function
  real(dp) function precision_score(y,p,average,pos_label) result(v)
    integer, intent(in) :: y(:),p(:)
    character(*), intent(in), optional :: average
    integer, intent(in), optional :: pos_label
    v=prf(y,p,1,average,pos_label)
  end function
  real(dp) function recall_score(y,p,average,pos_label) result(v)
    integer, intent(in) :: y(:),p(:)
    character(*), intent(in), optional :: average
    integer, intent(in), optional :: pos_label
    v=prf(y,p,2,average,pos_label)
  end function
  real(dp) function f1_score(y,p,average,pos_label) result(v)
    integer, intent(in) :: y(:),p(:)
    character(*), intent(in), optional :: average
    integer, intent(in), optional :: pos_label
    v=prf(y,p,3,average,pos_label)
  end function
  real(dp) function jaccard_score(y,p,average,pos_label) result(v)
    integer, intent(in) :: y(:),p(:)
    character(*), intent(in), optional :: average
    integer, intent(in), optional :: pos_label
    v=prf(y,p,4,average,pos_label)
  end function
  real(dp) function hamming_loss(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    v=1-accuracy_score(y,p)
  end function
  real(dp) function balanced_accuracy_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    integer, allocatable :: classes(:)
    integer :: i,n
    real(dp) :: r
    r=accuracy_score(y,p); classes=unique_labels(y); v=0
    do i=1,size(classes)
      n=count(y==classes(i)); v=v+real(count(y==classes(i).and.p==classes(i)),dp)/n
    end do
    v=v/size(classes)
  end function
  real(dp) function matthews_corrcoef(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    integer, allocatable :: cm(:,:)
    real(dp), allocatable :: a(:),b(:)
    real(dp) :: n,c,den
    integer :: i
    cm=confusion_matrix(y,p); a=real(sum(cm,dim=1),dp); b=real(sum(cm,dim=2),dp)
    n=sum(a); c=0
    do i=1,size(a)
      c=c+cm(i,i)
    end do
    den=sqrt(max(0._dp,(n*n-dot_product(a,a))*(n*n-dot_product(b,b))))
    v=0; if(den>0) v=(c*n-dot_product(a,b))/den
  end function
  real(dp) function cohen_kappa_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    integer, allocatable :: cm(:,:)
    real(dp) :: expected,observed,n
    cm=confusion_matrix(y,p); n=sum(cm)
    expected=dot_product(real(sum(cm,dim=1),dp),real(sum(cm,dim=2),dp))/(n*n)
    observed=accuracy_score(y,p)
    call require(expected<1,'kappa undefined for a single constant class')
    v=(observed-expected)/(1-expected)
  end function
  real(dp) function log_loss(y,prob,labels) result(v)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:,:)
    integer, intent(in), optional :: labels(:)
    integer, allocatable :: classes(:)
    integer :: i,k
    real(dp) :: row(size(prob,2))
    call check_xy(prob,size(y)); call require(all(prob>=0).and.all(prob<=1),'invalid probability')
    classes=unique_labels(y); if(present(labels)) classes=labels
    call require(size(classes)==size(prob,2),'probability class count mismatch')
    v=0
    do i=1,size(y)
      k=label_index(classes,y(i)); call require(k>0,'unknown log_loss target label')
      row=max(epsilon(1._dp),min(1-epsilon(1._dp),prob(i,:)))
      row=row/sum(row); v=v-log(row(k))
    end do
    v=v/size(y)
  end function
  real(dp) function roc_auc_score(y,scores,pos_label) result(v)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    integer, intent(in), optional :: pos_label
    integer :: i,j,pos,np,nn
    integer, allocatable :: classes(:)
    call require(size(y)==size(scores),'AUC size mismatch')
    classes=unique_labels(y); call require(size(classes)==2,'AUC requires two classes')
    pos=classes(2); if(present(pos_label)) pos=pos_label
    np=count(y==pos); nn=size(y)-np; call require(np>0.and.nn>0,'invalid positive label')
    v=0
    do i=1,size(y)
      if(y(i)/=pos) cycle
      do j=1,size(y)
        if(y(j)==pos) cycle
        if(scores(i)>scores(j)) v=v+1
        if(scores(i)==scores(j)) v=v+0.5_dp
      end do
    end do
    v=v/(real(np,dp)*nn)
  end function
  real(dp) function average_precision_score(y,scores,pos_label) result(v)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    integer, intent(in), optional :: pos_label
    integer :: order(size(y)),i,j,pos,tp,previous,np
    call require(size(y)>0.and.size(y)==size(scores),'AP size mismatch')
    pos=1; if(present(pos_label)) pos=pos_label
    order=argsort(-scores); np=count(y==pos); v=0; tp=0; i=1
    if(np==0) return
    do while(i<=size(y))
      j=i
      do while(j<size(y))
        if(scores(order(j+1))/=scores(order(i))) exit
        j=j+1
      end do
      previous=tp; tp=tp+count(y(order(i:j))==pos)
      v=v+real(tp-previous,dp)*tp/j; i=j+1
    end do
    v=v/np
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
  subroutine information(y,p,h,c)
    integer, intent(in) :: y(:),p(:)
    real(dp), intent(out) :: h,c
    real(dp), allocatable :: t(:,:),a(:),b(:)
    real(dp) :: ha,hb,mi,n
    integer :: i,j
    call contingency(y,p,t); a=sum(t,dim=2); b=sum(t,dim=1); n=size(y)
    ha=-sum(a/n*log(a/n)); hb=-sum(b/n*log(b/n)); mi=0
    do j=1,size(b)
      do i=1,size(a)
        if(t(i,j)>0) mi=mi+t(i,j)/n*log(t(i,j)*n/(a(i)*b(j)))
      end do
    end do
    h=1; c=1
    if(ha>0) h=mi/ha
    if(hb>0) c=mi/hb
  end subroutine
  real(dp) function homogeneity_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp) :: c
    call information(y,p,v,c)
  end function
  real(dp) function completeness_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp) :: h
    call information(y,p,h,v)
  end function
  real(dp) function v_measure_score(y,p) result(v)
    integer, intent(in) :: y(:),p(:)
    real(dp) :: h,c
    call information(y,p,h,c); v=0
    if(h+c>0) v=2*h*c/(h+c)
  end function
end module
