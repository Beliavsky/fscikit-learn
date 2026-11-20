! Native dense sklearn 1.6.1 spectral co-/biclustering. BSD provenance in THIRD_PARTY_NOTICES.md.
module sklearn_bicluster
  use sklearn_common, only: dp,require,check_x,dense_svd,argsort
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_minibatch_kmeans, only: mini_batch_kmeans
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: spectral_coclustering,spectral_biclustering
  type :: bicluster_base
    integer :: n_init=10,random_state=0,n_features_in_=0,n_rows_=0,n_biclusters_=0,n_normalization_iter_=0
    character(16) :: init='k-means++'
    logical :: mini_batch=.false.,normalization_converged_=.false.
    integer, allocatable :: row_labels_(:),column_labels_(:)
    logical, allocatable :: rows_(:,:),columns_(:,:)
    real(dp), allocatable :: normalized_data_(:,:),singular_values_(:),left_vectors_(:,:),right_vectors_(:,:)
  contains
    procedure :: get_indices
    procedure :: get_shape
    procedure :: get_submatrix
    procedure :: get_biclusters
  end type
  type, extends(bicluster_base) :: spectral_coclustering
    integer :: n_clusters=3,n_iter_=0
    real(dp) :: inertia_=0
    real(dp), allocatable :: row_scaling_(:),column_scaling_(:),embedding_(:,:)
  contains
    procedure :: fit=>coclustering_fit
  end type
  type, extends(bicluster_base) :: spectral_biclustering
    integer :: n_clusters=3,n_row_clusters=0,n_column_clusters=0,n_components=6,n_best=3
    character(16) :: method='bistochastic'
    integer, allocatable :: best_left_indices_(:),best_right_indices_(:)
    real(dp), allocatable :: left_piecewise_errors_(:),right_piecewise_errors_(:)
    real(dp), allocatable :: row_projection_(:,:),column_projection_(:,:)
  contains
    procedure :: fit=>biclustering_fit
  end type
contains
  subroutine controls(self,x)
    class(bicluster_base), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x)
    call require(self%n_init>=1,'spectral bicluster n_init must be positive')
    call require(self%random_state>=0,'spectral bicluster seed must be nonnegative')
    call require(self%init=='random'.or.self%init=='k-means++','unknown spectral bicluster initialization')
    self%n_rows_=size(x,1); self%n_features_in_=size(x,2)
  end subroutine
  function nonnegative(x,minimum) result(z)
    real(dp), intent(in) :: x(:,:),minimum
    real(dp) :: z(size(x,1),size(x,2))
    z=x+max(0._dp,minimum-minval(x))
    call require(all(ieee_is_finite(z)),'spectral bicluster data shift overflow')
  end function
  subroutine scale_normalize(x,z,row_scale,column_scale)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: z(:,:),row_scale(:),column_scale(:)
    real(dp) :: work(size(x,1),size(x,2)),rs(size(x,1)),cs(size(x,2))
    work=nonnegative(x,0._dp); rs=sum(work,dim=2); cs=sum(work,dim=1)
    call require(all(ieee_is_finite(rs)).and.all(ieee_is_finite(cs)),'spectral bicluster margin overflow')
    call require(all(rs>0).and.all(cs>0),'spectral bicluster normalization needs positive row and column sums')
    row_scale=1/sqrt(rs); column_scale=1/sqrt(cs)
    z=spread(row_scale,2,size(x,2))*work*spread(column_scale,1,size(x,1))
  end subroutine
  subroutine normalize(self,x,method)
    class(bicluster_base), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: old(:,:),updated(:,:),rs(:),cs(:)
    real(dp) :: row_mean(size(x,1)),col_mean(size(x,2)),mean
    integer :: it
    self%normalization_converged_=.true.; self%n_normalization_iter_=1
    select case(method)
    case('scale')
      call scale_normalize(x,self%normalized_data_,rs,cs)
    case('log')
      old=log(nonnegative(x,1._dp)); row_mean=sum(old,dim=2)/size(x,2)
      col_mean=sum(old,dim=1)/size(x,1); mean=sum(row_mean)/size(x,1)
      self%normalized_data_=old-spread(row_mean,2,size(x,2))-spread(col_mean,1,size(x,1))+mean
    case('bistochastic')
      old=nonnegative(x,0._dp); self%normalization_converged_=.false.
      do it=1,1000
        call scale_normalize(old,updated,rs,cs)
        if(sqrt(sum((old-updated)**2))<1.e-5_dp) then
          self%normalization_converged_=.true.; exit
        end if
        old=updated
      end do
      self%n_normalization_iter_=min(it,1000); self%normalized_data_=updated
    end select
  end subroutine
  subroutine complete_left(u)
    real(dp), intent(inout) :: u(:,:)
    real(dp) :: candidate(size(u,1)),best(size(u,1)),length,best_length
    integer :: j,i,k,pass
    do j=1,size(u,2)
      if(sum(u(:,j)**2)>.5_dp) cycle
      best_length=-1
      do i=1,size(u,1)
        candidate=0; candidate(i)=1
        do pass=1,2
          do k=1,size(u,2)
            if(k==j) cycle
            candidate=candidate-u(:,k)*dot_product(u(:,k),candidate)
          end do
        end do
        length=sum(candidate**2)
        if(length>best_length) then
          best_length=length; best=candidate
        end if
        ! At most 2*rank coordinates can have squared residual below one half.
        if(best_length>=.5_dp) exit
      end do
      call require(best_length>epsilon(1._dp)**2,'spectral bicluster null-space completion failed')
      u(:,j)=best/sqrt(best_length)
    end do
  end subroutine
  subroutine factor(self,requested,discard)
    class(bicluster_base), intent(inout) :: self
    integer(int64), intent(in) :: requested
    integer, intent(in) :: discard
    real(dp), allocatable :: u(:,:),v(:,:),s(:),tu(:,:),tv(:,:)
    integer :: m,n,r,keep,j,pivot
    m=self%n_rows_; n=self%n_features_in_; r=min(m,n); keep=int(min(requested,int(r,int64)))
    call require(keep>discard,'spectral bicluster has no singular vectors after discarding the leading vector')
    allocate(s(r))
    if(m>=n) then
      allocate(u(m,r),v(n,r)); call dense_svd(self%normalized_data_,s,u,v); call complete_left(u(:,:keep))
    else
      allocate(tu(n,r),tv(m,r)); call dense_svd(transpose(self%normalized_data_),s,tu,tv)
      call complete_left(tu(:,:keep)); u=tv; v=tu
    end if
    do j=1,r
      pivot=maxloc(abs(v(:,j)),dim=1)
      if(v(pivot,j)<0) then
        v(:,j)=-v(:,j); u(:,j)=-u(:,j)
      end if
    end do
    self%singular_values_=s(:keep)
    self%left_vectors_=u(:,discard+1:keep); self%right_vectors_=v(:,discard+1:keep)
  end subroutine
  function sample_unique(rng,n,k) result(ids)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: n,k
    integer :: ids(k),batch(k),found,needed,i,j,index
    real(dp) :: probability(n),cdf(n),value
    probability=1._dp/n; found=0
    do while(found<k)
      if(found>0) probability(ids(:found))=0
      cdf=probability
      do i=2,n
        cdf(i)=cdf(i-1)+cdf(i)
      end do
      cdf=cdf/cdf(n); needed=k-found
      do j=1,needed
        value=rng%uniform(); index=n
        do i=1,n
          if(value<cdf(i)) then
            index=i; exit
          end if
        end do
        batch(j)=index
      end do
      do j=1,needed
        if(any(batch(:j-1)==batch(j))) cycle
        found=found+1; ids(found)=batch(j)
      end do
    end do
  end function
  function seed_centers(x,k,method,rng) result(centers)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: k
    character(*), intent(in) :: method
    type(legacy_random_state), intent(inout) :: rng
    real(dp) :: centers(k,size(x,2)),dist(size(x,1)),cdf(size(x,1)),candidate(size(x,1)),best_dist(size(x,1))
    real(dp), allocatable :: u(:)
    real(dp) :: best,cost,potential
    integer :: ids(k),first(1),i,j,c,trials,chosen,best_id
    if(method=='random') then
      ids=sample_unique(rng,size(x,1),k); centers=x(ids,:); return
    end if
    first=sample_unique(rng,size(x,1),1); centers(1,:)=x(first(1),:)
    do i=1,size(dist)
      dist(i)=max(0._dp,sum(centers(1,:)**2)-2*dot_product(centers(1,:),x(i,:))+sum(x(i,:)**2))
    end do
    potential=sum(dist); trials=2+int(log(real(k,dp))); allocate(u(trials))
    do c=2,k
      cdf=dist
      do i=2,size(dist)
        cdf(i)=cdf(i-1)+cdf(i)
      end do
      do j=1,trials
        u(j)=rng%uniform()*potential
      end do
      best=huge(1._dp); best_id=1
      do j=1,trials
        chosen=size(dist)
        do i=1,size(dist)
          if(cdf(i)>=u(j)) then
            chosen=i; exit
          end if
        end do
        do i=1,size(dist)
          candidate(i)=min(dist(i),max(0._dp,sum(x(chosen,:)**2)-2*dot_product(x(chosen,:),x(i,:))+sum(x(i,:)**2)))
        end do
        cost=sum(candidate)
        if(cost<best) then
          best=cost; best_id=chosen; best_dist=candidate
        end if
      end do
      centers(c,:)=x(best_id,:); dist=best_dist; potential=best
    end do
  end function
  function assign(x,centers) result(labels)
    real(dp), intent(in) :: x(:,:),centers(:,:)
    integer :: labels(size(x,1)),i,j
    real(dp) :: norms(size(centers,1)),best,value
    norms=sum(centers**2,dim=2)
    do i=1,size(x,1)
      labels(i)=1; best=norms(1)-2*dot_product(x(i,:),centers(1,:))
      do j=2,size(centers,1)
        value=norms(j)-2*dot_product(x(i,:),centers(j,:))
        if(value<best) then
          best=value; labels(i)=j
        end if
      end do
    end do
  end function
  logical function same_partition(a,b,k) result(same)
    integer, intent(in) :: a(:),b(:),k
    integer :: mapping(k),i
    mapping=0; same=.true.
    do i=1,size(a)
      if(mapping(a(i))==0) mapping(a(i))=b(i)
      if(mapping(a(i))/=b(i)) then
        same=.false.; return
      end if
    end do
  end function
  subroutine cluster(self,data,k,centers,labels,cost,n_iter)
    class(bicluster_base), intent(in) :: self
    real(dp), intent(in) :: data(:,:)
    integer, intent(in) :: k
    real(dp), allocatable, intent(out) :: centers(:,:)
    integer, allocatable, intent(out) :: labels(:)
    real(dp), intent(out) :: cost
    integer, intent(out) :: n_iter
    type(mini_batch_kmeans) :: model
    type(legacy_random_state) :: rng
    real(dp) :: x(size(data,1),size(data,2)),mean(size(data,2)),tolerance,value,best
    real(dp) :: updated(k,size(data,2)),mass(k),distance(size(data,1)),shift(k)
    real(dp), allocatable :: current(:,:)
    integer :: current_labels(size(data,1)),old(size(data,1)),best_labels(size(data,1)),far,i,j,c,it,run,largest
    logical :: strict,replace_best
    call check_x(data); call require(k>0.and.k<=size(data,1),'spectral bicluster centroid count exceeds observations')
    if(self%mini_batch) then
      model%n_clusters=k; model%n_init=self%n_init; model%init=self%init; model%random_state=self%random_state
      call model%fit(data); centers=model%cluster_centers_; labels=model%labels_
      cost=model%inertia_; n_iter=model%n_iter_; return
    end if
    mean=sum(data,dim=1)/size(data,1); x=data-spread(mean,1,size(data,1))
    tolerance=1.e-4_dp*sum(sum(x**2,dim=1)/size(data,1))/size(data,2)
    call rng%seed(self%random_state); best=huge(1._dp)
    do run=1,self%n_init
      current=seed_centers(x,k,self%init,rng); old=0; strict=.false.
      do it=1,300
        current_labels=assign(x,current); updated=0; mass=0
        do i=1,size(x,1)
          c=current_labels(i); updated(c,:)=updated(c,:)+x(i,:); mass(c)=mass(c)+1
        end do
        if(any(mass==0)) then
          do i=1,size(x,1)
            distance(i)=sum((x(i,:)-current(current_labels(i),:))**2)
          end do
          if(maxval(distance)>0) then
            do j=1,k
              if(mass(j)/=0) cycle
              far=maxloc(distance,dim=1); c=current_labels(far)
              updated(c,:)=updated(c,:)-x(far,:); updated(j,:)=x(far,:)
              mass(c)=mass(c)-1; mass(j)=1; distance(far)=-1
            end do
          end if
        end if
        largest=maxloc(mass,dim=1)
        do j=1,k
          if(mass(j)>0) then
            updated(j,:)=updated(j,:)*(1/mass(j))
          else
            updated(j,:)=updated(largest,:)
          end if
          shift(j)=sqrt(sum((current(j,:)-updated(j,:))**2))
        end do
        current=updated
        if(all(current_labels==old)) then
          strict=.true.; exit
        end if
        if(sum(shift**2)<=tolerance) exit
        old=current_labels
      end do
      if(.not.strict) current_labels=assign(x,current)
      value=0
      do i=1,size(x,1)
        value=value+sum((x(i,:)-current(current_labels(i),:))**2)
      end do
      replace_best=run==1
      if(run>1) replace_best=value<best.and..not.same_partition(current_labels,best_labels,k)
      if(replace_best) then
        best=value; best_labels=current_labels; centers=current+spread(mean,1,k); n_iter=min(it,300)
      end if
    end do
    cost=best; labels=best_labels-1
  end subroutine
  subroutine coclustering_fit(self,x)
    class(spectral_coclustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: centers(:,:)
    integer, allocatable :: labels(:)
    integer :: requested,j,n,p
    integer(int64) :: power
    call controls(self,x); n=size(x,1); p=size(x,2)
    call require(self%n_clusters>=2.and.self%n_clusters<=n,'invalid SpectralCoclustering n_clusters (need 2 to n_rows)')
    call scale_normalize(x,self%normalized_data_,self%row_scaling_,self%column_scaling_)
    self%n_normalization_iter_=1; self%normalization_converged_=.true.
    power=1; requested=1
    do while(power<self%n_clusters)
      power=2*power; requested=requested+1
    end do
    call factor(self,int(requested,int64),1)
    self%embedding_=spread(spread(0._dp,1,n+p),2,size(self%left_vectors_,2))
    do j=1,size(self%left_vectors_,2)
      self%embedding_(:n,j)=self%row_scaling_*self%left_vectors_(:,j)
      self%embedding_(n+1:,j)=self%column_scaling_*self%right_vectors_(:,j)
    end do
    call cluster(self,self%embedding_,self%n_clusters,centers,labels,self%inertia_,self%n_iter_)
    self%row_labels_=labels(:n); self%column_labels_=labels(n+1:); self%n_biclusters_=self%n_clusters
    self%rows_=spread(spread(.false.,1,self%n_clusters),2,n)
    do j=1,self%n_clusters
      self%rows_(j,:)=self%row_labels_==j-1
    end do
    self%columns_=spread(spread(.false.,1,self%n_clusters),2,p)
    do j=1,self%n_clusters
      self%columns_(j,:)=self%column_labels_==j-1
    end do
  end subroutine
  subroutine best_piecewise(self,vectors,k,errors,indices)
    class(bicluster_base), intent(in) :: self
    real(dp), intent(in) :: vectors(:,:)
    integer, intent(in) :: k
    real(dp), allocatable, intent(out) :: errors(:)
    integer, allocatable, intent(out) :: indices(:)
    real(dp), allocatable :: centers(:,:)
    integer, allocatable :: labels(:)
    real(dp) :: cost
    integer :: i,it
    allocate(errors(size(vectors,2)))
    do i=1,size(vectors,2)
      call cluster(self,vectors(:,i:i),k,centers,labels,cost,it)
      errors(i)=sqrt(sum((vectors(:,i)-centers(labels+1,1))**2))
    end do
    indices=argsort(errors)-1
  end subroutine
  subroutine biclustering_fit(self,x)
    class(spectral_biclustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: centers(:,:)
    real(dp) :: cost
    integer :: kr,kc,discard,best,i,j,index,it
    call controls(self,x)
    kr=self%n_clusters; kc=self%n_clusters
    if(self%n_row_clusters/=0.or.self%n_column_clusters/=0) then
      call require(self%n_row_clusters>0.and.self%n_column_clusters>0,'spectral bicluster cluster overrides need a pair')
      kr=self%n_row_clusters; kc=self%n_column_clusters
    end if
    call require(kr>=1.and.kr<=size(x,1),'invalid SpectralBiclustering row clusters')
    call require(kc>=1.and.kc<=min(size(x,1),size(x,2)),'invalid SpectralBiclustering column clusters')
    call require(self%n_components>=1.and.self%n_best>=1,'spectral bicluster component counts must be positive')
    call require(self%n_best<=self%n_components,'spectral bicluster n_best exceeds n_components')
    call require(self%method=='scale'.or.self%method=='bistochastic'.or.self%method=='log', &
      'unknown spectral bicluster normalization')
    call normalize(self,x,self%method); discard=1
    if(self%method=='log') discard=0
    call factor(self,int(self%n_components,int64)+discard,discard)
    call best_piecewise(self,self%left_vectors_,kr,self%left_piecewise_errors_,self%best_left_indices_)
    call best_piecewise(self,self%right_vectors_,kc,self%right_piecewise_errors_,self%best_right_indices_)
    best=min(self%n_best,size(self%left_vectors_,2))
    self%best_left_indices_=self%best_left_indices_(:best); self%best_right_indices_=self%best_right_indices_(:best)
    self%row_projection_=matmul(x,self%right_vectors_(:,self%best_right_indices_+1))
    self%column_projection_=matmul(transpose(x),self%left_vectors_(:,self%best_left_indices_+1))
    call cluster(self,self%row_projection_,kr,centers,self%row_labels_,cost,it)
    call cluster(self,self%column_projection_,kc,centers,self%column_labels_,cost,it)
    call require(int(kr,int64)*kc<=huge(1),'spectral bicluster indicator count overflow')
    self%n_biclusters_=kr*kc
    self%rows_=spread(spread(.false.,1,kr*kc),2,size(x,1))
    self%columns_=spread(spread(.false.,1,kr*kc),2,size(x,2))
    do i=1,kr
      do j=1,kc
        index=(i-1)*kc+j
        self%rows_(index,:)=self%row_labels_==i-1; self%columns_(index,:)=self%column_labels_==j-1
      end do
    end do
  end subroutine
  subroutine checked_index(self,index,position)
    class(bicluster_base), intent(in) :: self
    integer, intent(in) :: index
    integer, intent(out) :: position
    call require(allocated(self%rows_),'spectral bicluster is not fitted')
    call require(index>=0.and.index<self%n_biclusters_,'spectral bicluster index out of range')
    position=index+1
  end subroutine
  subroutine get_indices(self,index,row_indices,column_indices)
    class(bicluster_base), intent(in) :: self
    integer, intent(in) :: index
    integer, allocatable, intent(out) :: row_indices(:),column_indices(:)
    integer :: position,i
    call checked_index(self,index,position)
    row_indices=pack([(i,i=1,self%n_rows_)],self%rows_(position,:))
    column_indices=pack([(i,i=1,self%n_features_in_)],self%columns_(position,:))
  end subroutine
  function get_shape(self,index) result(dimensions)
    class(bicluster_base), intent(in) :: self
    integer, intent(in) :: index
    integer :: dimensions(2),position
    call checked_index(self,index,position)
    dimensions=[count(self%rows_(position,:)),count(self%columns_(position,:))]
  end function
  function get_submatrix(self,index,data) result(values)
    class(bicluster_base), intent(in) :: self
    integer, intent(in) :: index
    real(dp), intent(in) :: data(:,:)
    real(dp), allocatable :: values(:,:)
    integer, allocatable :: rows(:),columns(:)
    call self%get_indices(index,rows,columns); call check_x(data)
    call require(all(shape(data)==[self%n_rows_,self%n_features_in_]),'spectral bicluster submatrix data shape mismatch')
    values=data(rows,columns)
  end function
  subroutine get_biclusters(self,rows,columns)
    class(bicluster_base), intent(in) :: self
    logical, allocatable, intent(out) :: rows(:,:),columns(:,:)
    call require(allocated(self%rows_),'spectral bicluster is not fitted')
    rows=self%rows_; columns=self%columns_
  end subroutine
end module
