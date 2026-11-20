! Shared unweighted sklearn-compatible Lloyd centroid helper for native estimators.
! BSD/NumPy provenance retained in THIRD_PARTY_NOTICES.md; no public estimator here.
module sklearn_centroid_core
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  private
  public :: fit_centroids,centroid_seeds
contains
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
  function centroid_seeds(x,k,method,rng) result(centers)
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
  subroutine fit_centroids(data,k,n_init,seed,method,centers,labels,cost,n_iter,random_state)
    real(dp), intent(in) :: data(:,:)
    integer, intent(in) :: k,n_init,seed
    character(*), intent(in) :: method
    type(legacy_random_state), optional, intent(inout) :: random_state
    real(dp), allocatable, intent(out) :: centers(:,:)
    integer, allocatable, intent(out) :: labels(:)
    real(dp), intent(out) :: cost
    integer, intent(out) :: n_iter
    type(legacy_random_state) :: rng
    real(dp) :: x(size(data,1),size(data,2)),mean(size(data,2)),tolerance,value,best
    real(dp) :: updated(k,size(data,2)),mass(k),distance(size(data,1)),shift(k)
    real(dp), allocatable :: current(:,:)
    integer :: current_labels(size(data,1)),old(size(data,1)),best_labels(size(data,1)),far,i,j,c,it,run,largest
    logical :: strict,replace_best
    call check_x(data); call require(k>0.and.k<=size(data,1),'centroid count exceeds observations')
    mean=sum(data,dim=1)/size(data,1); x=data-spread(mean,1,size(data,1))
    tolerance=1.e-4_dp*sum(sum(x**2,dim=1)/size(data,1))/size(data,2)
    call require(n_init>=1,'centroid n_init must be positive')
    call require(method=='random'.or.method=='k-means++','unknown centroid initialization')
    if(present(random_state)) then
      rng=random_state
    else
      call rng%seed(seed)
    end if
    best=huge(1._dp)
    do run=1,n_init
      current=centroid_seeds(x,k,method,rng); old=0; strict=.false.
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
    if(present(random_state)) random_state=rng
  end subroutine
end module
