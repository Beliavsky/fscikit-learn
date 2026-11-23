! sklearn 1.6.1 manifold/_barnes_hut_tsne.pyx and neighbors/_quad_tree.pyx (BSD).
! Serial float32 spatial-tree forces; see THIRD_PARTY_NOTICES.md.
module sklearn_tsne_bh
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: iso_fortran_env, only: real32,int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: tsne_bh_objective,tsne_bh_probabilities
  type :: cell
    integer :: parent=0,children(8)=0,point=0,depth=0,mass=0
    logical :: leaf=.true.
    real(real32) :: width=0,low(3)=0,high(3)=0,center(3)=0,bary(3)=0
  end type
contains
  subroutine tsne_bh_objective(position,p,degrees,angle,error,gradient,skip,compute_error,tree_indices,tree_values)
    real(dp), intent(in) :: position(:,:),angle
    type(csr_matrix), intent(in) :: p
    integer, intent(in) :: degrees
    real(dp), intent(out) :: error
    real(dp), allocatable, intent(out) :: gradient(:,:)
    integer, optional, intent(in) :: skip
    logical, optional, intent(in) :: compute_error
    integer, allocatable, optional, intent(out) :: tree_indices(:,:)
    real(dp), allocatable, optional, intent(out) :: tree_values(:,:)
    type(cell), allocatable :: cells(:),larger(:)
    real(real32), allocatable :: x(:,:),negative(:,:),positive(:,:)
    real(real32) :: lo(3),hi(3),pt(3),old(3),delta(3),dist2,exponent,df,theta,mult,size_,q,pij,term,cost
    real(dp) :: sumq,qz,coefficient
    integer :: n,d,i,j,k,c,start,used,capacity,summary_count
    logical :: calculate
    call check_x(position);n=size(position,1);d=size(position,2)
    call require(d>=1.and.d<=3,'Barnes-Hut t-SNE requires one to three components')
    call require(degrees>0,'Barnes-Hut degrees must be positive')
    call require(ieee_is_finite(angle),'Barnes-Hut angle must be finite')
    call require(angle>=0.and.angle<=1,'Barnes-Hut angle must be in [0,1]')
    call p%validate();call require(p%n_rows==n.and.p%n_cols==n,'Barnes-Hut probability shape mismatch')
    call require(all(ieee_is_finite(p%data)),'Barnes-Hut probabilities must be finite')
    call require(all(p%data>=0),'Barnes-Hut probabilities must be nonnegative')
    if(size(p%data)>0) call require(maxval(p%data)<=real(huge(1._real32),dp),'Barnes-Hut probabilities exceed float32 range')
    call require(maxval(abs(position))<sqrt(real(huge(1._real32),dp))/16,'Barnes-Hut positions exceed working range')
    start=0;if(present(skip)) start=skip
    call require(start>=0.and.start<=n,'invalid Barnes-Hut skipped point count')
    calculate=.true.;if(present(compute_error)) calculate=compute_error
    x=real(position,real32);allocate(negative(n,d),positive(n,d),gradient(n,d));negative=0;positive=0;gradient=0
    lo=0;hi=0
    do c=1,d
      lo(c)=minval(x(:,c));hi(c)=maxval(x(:,c))
      hi(c)=max(hi(c)*(1._real32+.001_real32*sign(1._real32,hi(c))),hi(c)+.001_real32)
    end do
    capacity=100;allocate(cells(capacity));used=1
    cells(1)%low=lo;cells(1)%high=hi;cells(1)%center=(lo+hi)/2._real32
    cells(1)%width=maxval((hi(:d)-lo(:d))**2)
    do i=1,n
      pt=0;pt(:d)=x(i,:);call insert(pt,i,1)
    end do
    theta=real(angle,real32)**2;df=real(degrees,real32);exponent=(df+1._real32)/2._real32
    sumq=0;summary_count=0
    do i=start+1,n
      pt=0;pt(:d)=x(i,:);call summarize(1)
    end do
    sumq=max(sumq,epsilon(1._dp));cost=0
    do i=start+1,n
      do k=p%indptr(i),p%indptr(i+1)-1
        j=p%indices(k);pij=real(p%data(k),real32);dist2=0
        do c=1,d
          delta(c)=x(i,c)-x(j,c);dist2=dist2+delta(c)*delta(c)
        end do
        q=df/(df+dist2)
        if(degrees/=1) q=q**exponent
        term=pij*q
        if(calculate) then
          q=real(real(q,dp)/sumq,real32)
          cost=real(real(cost,dp)+real(pij,dp)*log(real(max(pij,tiny(1._real32))/max(q,tiny(1._real32)),dp)),real32)
        end if
        do c=1,d
          positive(i,c)=positive(i,c)+term*delta(c)
        end do
      end do
    end do
    coefficient=2*(real(degrees,dp)+1)/degrees
    do i=start+1,n
      do c=1,d
        term=real(real(positive(i,c),dp)-real(negative(i,c),dp)/sumq,real32)
        gradient(i,c)=real(real(term*real(coefficient,real32),real32),dp)
      end do
    end do
    error=real(cost,dp)
    if(.not.calculate) error=ieee_value(0._dp,ieee_quiet_nan)
    if(present(tree_indices)) then
      allocate(tree_indices(used,13))
      do j=1,used
        tree_indices(j,:)=[cells(j)%parent,cells(j)%children,cells(j)%point,cells(j)%depth,cells(j)%mass, &
          merge(1,0,cells(j)%leaf)]
      end do
    end if
    if(present(tree_values)) then
      allocate(tree_values(used,4*d+1))
      do j=1,used
        tree_values(j,:)=[real(cells(j)%low(:d),dp),real(cells(j)%high(:d),dp), &
          real(cells(j)%center(:d),dp),real(cells(j)%bary(:d),dp),real(cells(j)%width,dp)]
      end do
    end if
  contains
    subroutine ensure_capacity()
      if(used<capacity) return
      call require(capacity<huge(1)/2,'Barnes-Hut tree exceeds integer capacity')
      allocate(larger(2*capacity));larger(:used)=cells(:used);call move_alloc(larger,cells);capacity=2*capacity
    end subroutine
    integer function slot(point,node) result(which)
      real(real32), intent(in) :: point(3)
      integer, intent(in) :: node
      integer :: axis
      which=0
      do axis=1,d
        which=2*which
        if(point(axis)>=cells(node)%center(axis)) which=which+1
      end do
      which=which+1
    end function
    integer function new_child(point,node,index,mass) result(child)
      real(real32), intent(in) :: point(3)
      integer, intent(in) :: node,index,mass
      integer :: axis,which
      call ensure_capacity();used=used+1;child=used
      cells(child)%parent=node;cells(child)%depth=cells(node)%depth+1
      call require(cells(child)%depth<512,'Barnes-Hut spatial subdivision did not separate points')
      which=slot(point,node);cells(node)%children(which)=child;cells(node)%leaf=.false.;cells(node)%point=0
      do axis=1,d
        if(point(axis)>=cells(node)%center(axis)) then
          cells(child)%low(axis)=cells(node)%center(axis);cells(child)%high(axis)=cells(node)%high(axis)
        else
          cells(child)%low(axis)=cells(node)%low(axis);cells(child)%high(axis)=cells(node)%center(axis)
        end if
        cells(child)%center(axis)=(cells(child)%low(axis)+cells(child)%high(axis))/2._real32
      end do
      cells(child)%width=maxval((cells(child)%high(:d)-cells(child)%low(:d))**2)
      cells(child)%bary=point;cells(child)%point=index;cells(child)%mass=mass
    end function
    recursive subroutine insert(point,index,node)
      real(real32), intent(in) :: point(3)
      integer, intent(in) :: index,node
      integer :: child,axis,mass,old_index
      real(real32) :: previous(3)
      mass=cells(node)%mass
      if(mass==0) then
        cells(node)%bary=point;cells(node)%mass=1;cells(node)%point=index;return
      end if
      if(.not.cells(node)%leaf) then
        do axis=1,d
          cells(node)%bary(axis)=(real(mass,real32)*cells(node)%bary(axis)+point(axis))/real(mass+1,real32)
        end do
        cells(node)%mass=mass+1;child=cells(node)%children(slot(point,node))
        if(child==0) then
          child=new_child(point,node,index,1)
        else
          call insert(point,index,child)
        end if
      else if(all(point(:d)==cells(node)%bary(:d))) then
        ! Tested sklearn binary exports EPSILON=0, despite its pxd initializer.
        cells(node)%mass=mass+1
      else
        previous=cells(node)%bary;old_index=cells(node)%point
        child=new_child(previous,node,old_index,mass);call insert(point,index,node)
      end if
    end subroutine
    recursive subroutine summarize(node)
      integer, intent(in) :: node
      real(real32) :: difference(3),squared,weight,ratio
      real(dp) :: probability
      integer :: axis,child,which
      difference=pt-cells(node)%bary;squared=0
      do axis=1,d
        squared=squared+difference(axis)*difference(axis)
      end do
      if(cells(node)%leaf.and.all(difference(:d)==0)) return
      ratio=huge(1._real32)
      if(squared>0) then
        if(real(cells(node)%width,dp)/real(squared,dp)<=real(huge(1._real32),dp)) &
          ratio=cells(node)%width/squared
      end if
      if(cells(node)%leaf.or.ratio<theta) then
        probability=real(df/(df+squared),dp)
        if(degrees/=1) probability=probability**real(exponent,dp)
        weight=real(cells(node)%mass,real32);sumq=sumq+real(weight,dp)*probability
        mult=real(real(weight,dp)*probability*probability,real32)
        do axis=1,d
          negative(i,axis)=negative(i,axis)+mult*difference(axis)
        end do
        summary_count=summary_count+1
      else
        do which=1,2**d
          child=cells(node)%children(which)
          if(child/=0) call summarize(child)
        end do
      end if
    end subroutine
  end subroutine
  subroutine tsne_bh_probabilities(distances,perplexity,p)
    type(csr_matrix), intent(in) :: distances
    real(dp), intent(in) :: perplexity
    type(csr_matrix), intent(out) :: p
    real(dp), allocatable :: values(:),conditional(:),buffer(:),rows(:)
    integer, allocatable :: columns(:),counts(:),ptr(:),cursor(:),outcols(:)
    real(dp) :: beta,lower,upper,mass,moment,h,target,total,value
    integer :: n,k,i,j,t,lo,hi,step,at,key,used,edge,count,last
    logical :: has_lower,has_upper
    call distances%validate();n=distances%n_rows
    call require(n>=2.and.distances%n_cols==n,'Barnes-Hut distances must be square with at least two samples')
    call require(all(ieee_is_finite(distances%data)),'Barnes-Hut distances must be finite')
    call require(all(distances%data>=0),'Barnes-Hut distances must be nonnegative')
    if(size(distances%data)>0) call require(maxval(distances%data)<=real(huge(1._real32),dp), &
      'Barnes-Hut distances exceed float32 range')
    call require(ieee_is_finite(perplexity),'t-SNE perplexity must be finite')
    call require(perplexity>0.and.perplexity<n,'t-SNE requires 0 < perplexity < n_samples')
    call require(real(perplexity,real32)>0,'t-SNE perplexity is below float32 range')
    call require(size(distances%data,kind=int64)<huge(1)/2,'Barnes-Hut graph exceeds integer capacity')
    k=distances%indptr(2)-distances%indptr(1)
    call require(all(distances%indptr(2:)-distances%indptr(:n)==k),'Barnes-Hut neighbor rows must have equal length')
    values=real(real(distances%data,real32),dp);columns=distances%indices
    allocate(conditional(size(values)));conditional=0;target=log(real(real(perplexity,real32),dp))
    do i=1,n
      lo=distances%indptr(i);hi=distances%indptr(i+1)-1
      do j=lo+1,hi
        key=columns(j);value=values(j);at=j-1
        do while(at>=lo)
          if(columns(at)<=key) exit
          columns(at+1)=columns(at);values(at+1)=values(at);at=at-1
        end do
        columns(at+1)=key;values(at+1)=value
      end do
      beta=1;lower=0;upper=0;has_lower=.false.;has_upper=.false.
      do step=1,100
        mass=0
        do j=lo,hi
          if(k==n.and.j-lo+1==i) cycle
          conditional(j)=exp(-values(j)*beta);mass=mass+conditional(j)
        end do
        if(mass==0) mass=real(1.e-8_real32,dp)
        moment=0
        do j=lo,hi
          conditional(j)=conditional(j)/mass;moment=moment+values(j)*conditional(j)
        end do
        h=log(mass)+beta*moment
        if(abs(h-target)<=real(1.e-5_real32,dp)) exit
        if(h>target) then
          lower=beta;has_lower=.true.
          if(has_upper) then
            beta=(beta+upper)/2
          else
            beta=beta*2
          end if
        else
          upper=beta;has_upper=.true.
          if(has_lower) then
            beta=(beta+lower)/2
          else
            beta=beta/2
          end if
        end if
      end do
    end do
    allocate(counts(n),ptr(n+1),cursor(n));counts=0
    do i=1,n
      do j=distances%indptr(i),distances%indptr(i+1)-1
        if(conditional(j)==0) cycle
        counts(i)=counts(i)+1;counts(columns(j))=counts(columns(j))+1
      end do
    end do
    ptr(1)=1
    do i=1,n
      ptr(i+1)=ptr(i)+counts(i)
    end do
    allocate(outcols(ptr(n+1)-1),buffer(ptr(n+1)-1));cursor=ptr(:n)
    do i=1,n
      do j=distances%indptr(i),distances%indptr(i+1)-1
        if(conditional(j)==0) cycle
        t=columns(j);edge=cursor(i);outcols(edge)=t;buffer(edge)=conditional(j);cursor(i)=edge+1
        edge=cursor(t);outcols(edge)=i;buffer(edge)=conditional(j);cursor(t)=edge+1
      end do
    end do
    p%n_rows=n;p%n_cols=n;allocate(p%indptr(n+1),rows(n));p%indptr(1)=1;used=0
    do i=1,n
      lo=ptr(i);hi=ptr(i+1)-1
      do j=lo+1,hi
        key=outcols(j);value=buffer(j);at=j-1
        do while(at>=lo)
          if(outcols(at)<=key) exit
          outcols(at+1)=outcols(at);buffer(at+1)=buffer(at);at=at-1
        end do
        outcols(at+1)=key;buffer(at+1)=value
      end do
      last=0;rows(i)=0
      do j=lo,hi
        key=outcols(j);value=buffer(j)
        if(key==last) then
          buffer(used)=buffer(used)+value
        else
          used=used+1;outcols(used)=key;buffer(used)=value;last=key
        end if
      end do
      p%indptr(i+1)=used+1
      do j=p%indptr(i),p%indptr(i+1)-1
        rows(i)=rows(i)+buffer(j)
      end do
    end do
    total=max(numpy_sum(rows),epsilon(1._dp));p%indices=outcols(:used);p%data=buffer(:used)/total
  end subroutine
  recursive real(dp) function numpy_sum(a) result(value)
    real(dp), intent(in) :: a(:)
    real(dp) :: b(8)
    integer :: n,i,whole,split
    n=size(a);value=-0._dp
    if(n<8) then
      do i=1,n
        value=value+a(i)
      end do
    else if(n<=128) then
      b=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        b=b+a(i:i+7)
      end do
      value=((b(1)+b(2))+(b(3)+b(4)))+((b(5)+b(6))+(b(7)+b(8)))
      do i=whole+1,n
        value=value+a(i)
      end do
    else
      split=n/2;split=split-mod(split,8);value=numpy_sum(a(:split))+numpy_sum(a(split+1:))
    end if
  end function
end module

