! Dense and CSR-precomputed sklearn 1.6.1 HDBSCAN brute-backend contracts.
! Leland McInnes et al. BSD notices retained in LICENSES/hdbscan-BSD.md and THIRD_PARTY_NOTICES.md.
module sklearn_hdbscan
  use sklearn_common, only: dp,require,argsort
  use sklearn_sparse, only: csr_matrix
  use sklearn_stateless_preprocessing, only: sorted_sparse
  use sklearn_numpy_ordering, only: numpy_argsort
  use sklearn_ieee_math, only: ieee_quotient,ieee_product
  use sklearn_metrics, only: pairwise_distances
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_positive_inf,ieee_quiet_nan
  implicit none
  private
  public :: hdbscan
  type :: hdbscan
    integer :: min_cluster_size=5,min_samples=0,max_cluster_size=0,n_features_in_=0,n_clusters_=0
    real(dp) :: alpha=1,cluster_selection_epsilon=0,p=2,max_distance=0
    character(16) :: metric='euclidean',cluster_selection_method='eom',store_centers='none'
    logical :: allow_single_cluster=.false.,has_infinite_edges_=.false.
    integer, allocatable :: labels_(:),finite_indices_(:),children_(:,:),cluster_sizes_(:)
    real(dp), allocatable :: probabilities_(:),distances_(:),core_distances_(:),centroids_(:,:),medoids_(:,:)
    integer, allocatable :: mst_edges_(:,:),condensed_parents_(:),condensed_children_(:),condensed_sizes_(:)
    real(dp), allocatable :: mst_distances_(:),condensed_lambdas_(:),stabilities_(:)
    integer, allocatable :: selected_clusters_(:)
  contains
    procedure, private :: fit_dense=>fit,fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure, private :: fit_predict_dense=>fit_predict,fit_predict_csr
    generic :: fit_predict=>fit_predict_dense,fit_predict_csr
    procedure :: dbscan_clustering
  end type
contains
  real(dp) function infinity() result(value)
    value=ieee_value(0._dp,ieee_positive_inf)
  end function
  real(dp) function inverse_distance(value) result(result)
    real(dp), intent(in) :: value
    if(value==0) then
      result=infinity()
    else if(value<1/huge(1._dp)) then
      result=infinity()
    else
      result=1/value
    end if
  end function
  integer function find_root(parent,node) result(root)
    integer, intent(inout) :: parent(:)
    integer, intent(in) :: node
    integer :: current,next
    root=node
    do while(parent(root)/=root)
      root=parent(root)
    end do
    current=node
    do while(parent(current)/=current)
      next=parent(current); parent(current)=root; current=next
    end do
  end function
  subroutine union_rank(parent,ranks,a,b)
    integer, intent(inout) :: parent(:),ranks(:)
    integer, intent(in) :: a,b
    integer :: left,right
    left=find_root(parent,a); right=find_root(parent,b)
    if(left==right) return
    if(ranks(left)<ranks(right)) then
      parent(left)=right
    else if(ranks(left)>ranks(right)) then
      parent(right)=left
    else
      parent(right)=left; ranks(left)=ranks(left)+1
    end if
  end subroutine
  recursive real(dp) function kth_value(values,k) result(value)
    real(dp), intent(in) :: values(:)
    integer, intent(in) :: k
    real(dp) :: work(size(values)),pivot,temp
    integer :: left,right,i,j
    if(any(ieee_is_nan(values))) then
      if(k>count(.not.ieee_is_nan(values))) then
        value=ieee_value(0._dp,ieee_quiet_nan)
      else
        value=kth_value(pack(values,.not.ieee_is_nan(values)),k)
      end if
      return
    end if
    work=values; left=1; right=size(values)
    do while(left<right)
      pivot=work(k); i=left; j=right
      do
        do while(i<=right)
          if(work(i)>=pivot) exit
          i=i+1
        end do
        do while(j>=left)
          if(work(j)<=pivot) exit
          j=j-1
        end do
        if(i>j) exit
        temp=work(i); work(i)=work(j); work(j)=temp; i=i+1; j=j-1
        if(i>j) exit
      end do
      if(k<=j) then
        right=j
      else if(k>=i) then
        left=i
      else
        exit
      end if
    end do
    value=work(k)
  end function
  subroutine controls(self,n,p)
    class(hdbscan), intent(in) :: self
    integer, intent(in) :: n,p
    call require(n>0.and.p>0,'HDBSCAN X must be nonempty')
    call require(int(n,int64)*2<=huge(1),'HDBSCAN tree index capacity exceeded')
    call require(self%min_cluster_size>=2,'HDBSCAN min_cluster_size must be at least two')
    call require(self%min_samples>=0,'HDBSCAN min_samples must be positive or zero for auto')
    call require(self%max_cluster_size>=0,'HDBSCAN max_cluster_size must be positive or zero for none')
    call require(ieee_is_finite(self%alpha).and.ieee_is_finite(self%cluster_selection_epsilon).and. &
      ieee_is_finite(self%p),'HDBSCAN numeric controls must be finite')
    call require(self%alpha>0,'HDBSCAN alpha must be positive')
    call require(self%cluster_selection_epsilon>=0,'HDBSCAN selection epsilon must be nonnegative')
    call require(self%p>=1,'HDBSCAN Minkowski p must be at least one')
    call require(self%cluster_selection_method=='eom'.or.self%cluster_selection_method=='leaf', &
      'unknown HDBSCAN cluster selection method')
    call require(any(self%metric==[character(16)::'euclidean','sqeuclidean','manhattan','cityblock','chebyshev', &
      'minkowski','cosine','precomputed']),'unknown HDBSCAN metric')
    call require(any(self%store_centers==[character(16)::'none','centroid','medoid','both']),'unknown HDBSCAN center storage')
    if(self%metric=='precomputed') call require(self%store_centers=='none','HDBSCAN cannot store precomputed centers')
  end subroutine
  subroutine make_hierarchy(self,distances,children,heights,sizes)
    class(hdbscan), intent(inout) :: self
    real(dp), intent(inout) :: distances(:,:)
    integer, allocatable, intent(out) :: children(:,:),sizes(:)
    real(dp), allocatable, intent(out) :: heights(:)
    real(dp) :: keys(size(distances,1))
    integer :: n,i,j,current,next,left,right,new_node
    integer :: parent(2*size(distances,1)-1),node_sizes(2*size(distances,1)-1)
    integer, allocatable :: edges(:,:),order(:)
    logical :: remaining(size(distances,1))
    n=size(distances,1); self%core_distances_=spread(0._dp,1,n)
    j=self%min_samples
    if(j==0) j=self%min_cluster_size
    do i=1,n
      self%core_distances_(i)=kth_value(distances(i,:),j)
    end do
    do j=1,n
      do i=1,n
        distances(i,j)=max(distances(i,j),self%core_distances_(i),self%core_distances_(j))
      end do
    end do
    allocate(edges(n-1,2)); self%mst_distances_=spread(0._dp,1,n-1)
    keys=infinity(); current=1; remaining=.true.
    do i=1,n-1
      remaining(current)=.false.; next=0
      do j=1,n
        if(.not.remaining(j)) cycle
        keys(j)=min(keys(j),distances(current,j))
        if(next==0) then
          next=j
        else if(keys(j)<keys(next)) then
          next=j
        end if
      end do
      edges(i,:)=[current,next]; self%mst_distances_(i)=keys(next); current=next
    end do
    self%mst_edges_=reshape(self%finite_indices_(reshape(edges,[size(edges)])),[n-1,2])
    self%has_infinite_edges_=any(.not.ieee_is_finite(self%mst_distances_))
    order=argsort(self%mst_distances_); allocate(children(n-1,2),heights(n-1),sizes(n-1))
    parent=[(i,i=1,2*n-1)]; node_sizes=1
    do i=1,n-1
      left=find_root(parent,edges(order(i),1)); right=find_root(parent,edges(order(i),2)); new_node=n+i
      children(i,:)=[left,right]; heights(i)=self%mst_distances_(order(i)); sizes(i)=node_sizes(left)+node_sizes(right)
      parent(left)=new_node; parent(right)=new_node; node_sizes(new_node)=sizes(i)
    end do
  end subroutine
  subroutine condense(self,children,heights,sizes,n)
    class(hdbscan), intent(inout) :: self
    integer, intent(in) :: children(:,:),sizes(:),n
    real(dp), intent(in) :: heights(:)
    integer :: relabel(2*n-1),queue(2*n-1),local(2*n-1),length,head,tail,node,left,right,lc,rc,next,count,j,q
    integer :: parents(2*n),descendants(2*n),counts(2*n)
    real(dp) :: lambdas(2*n),lambda
    logical :: ignored(2*n-1)
    relabel=0; relabel(2*n-1)=n+1; next=n+2; count=0; ignored=.false.
    queue(1)=2*n-1; head=1; tail=1
    do while(head<=tail)
      node=queue(head); head=head+1
      if(node>n) then
        queue(tail+1:tail+2)=children(node-n,:); tail=tail+2
      end if
    end do
    length=tail
    do q=1,length
      node=queue(q)
      if(ignored(node).or.node<=n) cycle
      left=children(node-n,1); right=children(node-n,2); lc=1; rc=1
      if(left>n) lc=sizes(left-n)
      if(right>n) rc=sizes(right-n)
      lambda=inverse_distance(heights(node-n))
      if(lc>=self%min_cluster_size.and.rc>=self%min_cluster_size) then
        relabel(left)=next; next=next+1; relabel(right)=next; next=next+1
        call append(relabel(node),relabel(left),lambda,lc)
        call append(relabel(node),relabel(right),lambda,rc)
      else if(lc<self%min_cluster_size.and.rc<self%min_cluster_size) then
        call prune(left,relabel(node),lambda); call prune(right,relabel(node),lambda)
      else if(lc<self%min_cluster_size) then
        relabel(right)=relabel(node); call prune(left,relabel(node),lambda)
      else
        relabel(left)=relabel(node); call prune(right,relabel(node),lambda)
      end if
    end do
    self%condensed_parents_=parents(:count); self%condensed_children_=descendants(:count)
    self%condensed_lambdas_=lambdas(:count); self%condensed_sizes_=counts(:count)
  contains
    subroutine append(parent,child,value,size_cluster)
      integer, intent(in) :: parent,child,size_cluster
      real(dp), intent(in) :: value
      count=count+1; call require(count<=size(parents),'HDBSCAN condensed edge capacity exceeded')
      parents(count)=parent; descendants(count)=child; lambdas(count)=value; counts(count)=size_cluster
    end subroutine
    subroutine prune(root,parent,value)
      integer, intent(in) :: root,parent
      real(dp), intent(in) :: value
      integer :: current,h,t
      local(1)=root; h=1; t=1
      do while(h<=t)
        current=local(h); h=h+1; ignored(current)=.true.
        if(current<=n) then
          call append(parent,current,value,1)
        else
          local(t+1:t+2)=children(current-n,:); t=t+2
        end if
      end do
    end subroutine
  end subroutine
  subroutine select_clusters(self,n,labels,probabilities)
    class(hdbscan), intent(inout) :: self
    integer, intent(in) :: n
    integer, allocatable, intent(out) :: labels(:)
    real(dp), allocatable, intent(out) :: probabilities(:)
    integer :: root,last,i,node,child,parent,number,limit,top,target
    integer, allocatable :: ancestry(:),sizes(:),owners(:),mapping(:),first_child(:),next_sibling(:)
    real(dp), allocatable :: birth(:),death(:),stability(:),point_lambda(:)
    logical, allocatable :: choice(:),selected(:),blocked(:),merged(:),processed(:)
    real(dp) :: value,subtree,threshold
    root=n+1; last=maxval(self%condensed_parents_)
    allocate(ancestry(last),sizes(last),owners(last),mapping(last),first_child(last),next_sibling(last))
    allocate(birth(last),death(last),stability(last),point_lambda(n))
    allocate(choice(last),selected(last),blocked(last),merged(last),processed(last))
    ancestry=0; sizes=0; birth=0; death=0; stability=0; first_child=0; next_sibling=0
    do i=1,size(self%condensed_parents_)
      parent=self%condensed_parents_(i); child=self%condensed_children_(i); value=self%condensed_lambdas_(i)
      ancestry(child)=parent; birth(child)=value; sizes(child)=self%condensed_sizes_(i)
      death(parent)=max(death(parent),value)
      if(child<root) then
        point_lambda(child)=value
      else
        next_sibling(child)=first_child(parent); first_child(parent)=child
        if(parent==root) sizes(root)=sizes(root)+self%condensed_sizes_(i)
      end if
    end do
    do i=1,size(self%condensed_parents_)
      parent=self%condensed_parents_(i); value=self%condensed_lambdas_(i)
      if(.not.ieee_is_finite(value).and..not.ieee_is_finite(birth(parent))) then
        stability(parent)=ieee_value(0._dp,ieee_quiet_nan)
      else if(.not.ieee_is_nan(stability(parent))) then
        stability(parent)=stability(parent)+(value-birth(parent))*self%condensed_sizes_(i)
      end if
    end do
    self%stabilities_=stability(root:last); choice=.false.; selected=.false.; blocked=.false.
    top=root+1
    if(self%allow_single_cluster) top=root
    limit=n+1
    if(self%max_cluster_size>0) limit=self%max_cluster_size
    if(self%cluster_selection_method=='eom') then
      do node=last,top,-1
        subtree=0; child=first_child(node)
        do while(child/=0)
          if(ieee_is_nan(stability(child))) then
            subtree=ieee_value(0._dp,ieee_quiet_nan); exit
          end if
          subtree=subtree+stability(child); child=next_sibling(child)
        end do
        choice(node)=.true.
        if(sizes(node)>limit) then
          choice(node)=.false.
        else if(.not.ieee_is_nan(subtree).and..not.ieee_is_nan(stability(node))) then
          if(subtree>stability(node)) choice(node)=.false.
        end if
        if(.not.choice(node)) stability(node)=subtree
      end do
      do node=root,last
        if(node>root) blocked(node)=blocked(ancestry(node)).or.selected(ancestry(node))
        selected(node)=choice(node).and..not.blocked(node)
      end do
    else
      do node=root+1,last
        selected(node)=first_child(node)==0
      end do
    end if
    if(self%cluster_selection_epsilon>0.and.last>root) then
      number=count(selected); merged=.false.; processed=.false.
      if(number==1.and.selected(root)) then
        merged(root)=self%allow_single_cluster
      else
        do node=root+1,last
          if(.not.selected(node)) cycle
          target=node
          if(inverse_distance(birth(node))<self%cluster_selection_epsilon) then
            if(processed(node)) cycle
            do
              parent=ancestry(target)
              if(parent==root) then
                if(self%allow_single_cluster) target=root
                exit
              end if
              target=parent
              if(inverse_distance(birth(parent))>self%cluster_selection_epsilon) exit
            end do
            do child=root+1,last
              parent=child
              do while(parent/=0.and.parent/=target)
                parent=ancestry(parent)
              end do
              if(parent==target.and.child/=target) processed(child)=.true.
            end do
          end if
          merged(target)=.true.
        end do
      end if
      selected=merged
    end if
    self%selected_clusters_=pack([(i,i=1,last)],selected); number=size(self%selected_clusters_)
    mapping=-1; owners=0; owners(root)=root
    do i=1,number
      mapping(self%selected_clusters_(i))=i-1
    end do
    do node=root+1,last
      owners(node)=owners(ancestry(node))
      if(selected(node)) owners(node)=node
    end do
    allocate(labels(n),probabilities(n)); labels=-1; probabilities=0
    threshold=death(root)
    if(self%cluster_selection_epsilon>0) threshold=inverse_distance(self%cluster_selection_epsilon)
    do i=1,n
      node=owners(ancestry(i))
      if(node/=root) then
        labels(i)=mapping(node)
      else if(number==1.and.self%allow_single_cluster.and.selected(root)) then
        if(point_lambda(i)>=threshold) labels(i)=mapping(root)
      end if
      if(labels(i)<0) cycle
      if(death(node)==0.or..not.ieee_is_finite(point_lambda(i))) then
        probabilities(i)=1
      else
        probabilities(i)=min(point_lambda(i),death(node))/death(node)
      end if
    end do
    ! Compact occupied labels: an epsilon-boundary root can be selected but empty.
    mapping=-1; number=0
    do node=0,size(self%selected_clusters_)-1
      if(.not.any(labels==node)) cycle
      mapping(node+1)=number; number=number+1
    end do
    do i=1,n
      if(labels(i)>=0) labels(i)=mapping(labels(i)+1)
    end do
  end subroutine
  subroutine fit(self,x)
    class(hdbscan), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: data(:,:),distances(:,:),heights(:),probabilities(:)
    integer, allocatable :: markers(:),children(:,:),sizes(:),labels(:)
    logical :: has_positive,has_negative
    integer :: n,nf,i,j,k,row,left,right,next
    real(dp) :: difference
    call controls(self,size(x,1),size(x,2)); n=size(x,1); self%n_features_in_=size(x,2)
    allocate(markers(n)); markers=0
    if(self%metric=='precomputed') then
      call require(size(x,2)==n,'HDBSCAN precomputed distances must be square')
      call require(.not.any(ieee_is_nan(x)),'HDBSCAN precomputed distances contain NaN')
      call require(all(x>=0),'HDBSCAN precomputed distances must be nonnegative')
      do j=1,n
        do i=j+1,n
          if(ieee_is_finite(x(i,j)).and.ieee_is_finite(x(j,i))) then
            difference=abs(x(i,j)-x(j,i))
            call require(difference<=1.e-8_dp+1.e-5_dp*min(abs(x(j,i)),abs(x(i,j))), &
              'HDBSCAN precomputed distances must be symmetric')
          else
            call require(.not.ieee_is_finite(x(i,j)).and..not.ieee_is_finite(x(j,i)), &
              'HDBSCAN precomputed distances must be symmetric')
          end if
        end do
      end do
      distances=x
    else
      do i=1,n
        if(any(ieee_is_nan(x(i,:)))) then
          markers(i)=-3
        else if(.not.all(ieee_is_finite(x(i,:)))) then
          has_positive=.false.; has_negative=.false.
          do j=1,size(x,2)
            if(ieee_is_finite(x(i,j))) cycle
            if(x(i,j)>0) has_positive=.true.
            if(x(i,j)<0) has_negative=.true.
          end do
          markers(i)=-2
          if(has_positive.and.has_negative) markers(i)=-3
        end if
      end do
    end if
    self%finite_indices_=pack([(i,i=1,n)],markers==0); nf=size(self%finite_indices_)
    call require(nf>=2,'HDBSCAN requires at least two finite samples')
    k=self%min_samples
    if(k==0) k=self%min_cluster_size
    call require(k<=nf,'HDBSCAN min_samples exceeds finite sample count')
    if(self%metric/='precomputed') then
      data=x(self%finite_indices_,:); distances=pairwise_distances(data,data,self%metric,self%p)
      do i=1,nf
        distances(i,i)=0
      end do
    end if
    call require(.not.any(ieee_is_nan(distances)),'HDBSCAN distance calculation produced NaN')
    if(self%alpha<1) call require(maxval(distances,mask=ieee_is_finite(distances))<=huge(1._dp)*self%alpha, &
      'HDBSCAN distance scaling overflow')
    distances=distances/self%alpha
    call make_hierarchy(self,distances,children,heights,sizes)
    deallocate(distances)
    if(allocated(data)) deallocate(data)
    call condense(self,children,heights,sizes,nf)
    call select_clusters(self,nf,labels,probabilities)
    self%labels_=markers; self%probabilities_=spread(0._dp,1,n)
    self%labels_(self%finite_indices_)=labels; self%probabilities_(self%finite_indices_)=probabilities
    do i=1,n
      if(markers(i)==-3) self%probabilities_(i)=ieee_value(0._dp,ieee_quiet_nan)
    end do
    self%n_clusters_=max(0,maxval(labels)+1)
    self%children_=spread(spread(0,1,n-1),2,2); self%distances_=spread(infinity(),1,n-1)
    self%cluster_sizes_=spread(0,1,n-1)
    do i=1,nf-1
      do j=1,2
        if(children(i,j)<=nf) then
          self%children_(i,j)=self%finite_indices_(children(i,j))
        else
          self%children_(i,j)=children(i,j)+n-nf
        end if
      end do
      self%distances_(i)=heights(i); self%cluster_sizes_(i)=sizes(i)
    end do
    next=n+nf-1; row=nf-1
    do i=1,n
      if(markers(i)==0) cycle
      row=row+1; self%children_(row,:)=[i,next]; self%cluster_sizes_(row)=nf+row-(nf-1); next=next+1
    end do
    if(allocated(self%centroids_)) deallocate(self%centroids_)
    if(allocated(self%medoids_)) deallocate(self%medoids_)
    if(self%store_centers/='none') call centers(self,x)
  end subroutine
  function stable_edge_order(values) result(order)
    real(dp), intent(in) :: values(:)
    integer, allocatable :: order(:),work(:)
    integer :: n,width,first,middle,last,a,b,t,i
    n=size(values); order=[(i,i=1,n)]; allocate(work(n)); width=1
    do while(width<n)
      first=1
      do while(first<=n)
        middle=first+min(width,n-first+1); last=middle+min(width,n-middle+1)-1
        a=first; b=middle
        do t=first,last
          if(a>=middle) then
            work(t)=order(b); b=b+1
          else if(b>last) then
            work(t)=order(a); a=a+1
          else if(edge_before(values(order(a)),values(order(b)))) then
            work(t)=order(a); a=a+1
          else
            work(t)=order(b); b=b+1
          end if
        end do
        first=last+1
      end do
      order=work
      if(width>n/2) exit
      width=2*width
    end do
  end function
  logical function edge_before(a,b) result(before)
    real(dp), intent(in) :: a,b
    if(ieee_is_nan(b)) then
      before=.true.
    else if(ieee_is_nan(a)) then
      before=.false.
    else
      before=a<=b
    end if
  end function
  subroutine sparse_symmetry(graph)
    type(csr_matrix), intent(in) :: graph
    integer :: i,e,j,first,last,middle,reverse
    real(dp) :: a,b
    do i=1,graph%n_rows
      do e=graph%indptr(i),graph%indptr(i+1)-1
        j=graph%indices(e); first=graph%indptr(j); last=graph%indptr(j+1)-1; reverse=0
        do while(first<=last)
          middle=first+(last-first)/2
          if(graph%indices(middle)<i) then
            first=middle+1
          else if(graph%indices(middle)>i) then
            last=middle-1
          else
            reverse=middle; exit
          end if
        end do
        call require(reverse>0,'HDBSCAN sparse precomputed structure must be symmetric')
        a=graph%data(e); b=graph%data(reverse)
        if(ieee_is_finite(a).and.ieee_is_finite(b)) then
          call require(abs(a-b)<=1.e-9_dp+1.e-7_dp*min(abs(a),abs(b)), &
            'HDBSCAN sparse precomputed values must be symmetric')
        else
          call require(a==b,'HDBSCAN sparse precomputed values must be symmetric')
        end if
      end do
    end do
  end subroutine
  subroutine sparse_hierarchy(self,graph,children,heights,sizes)
    class(hdbscan), intent(inout) :: self
    type(csr_matrix), intent(inout) :: graph
    integer, allocatable, intent(out) :: children(:,:),sizes(:)
    real(dp), allocatable, intent(out) :: heights(:)
    integer, allocatable :: rows(:),parent(:),ranks(:),order(:),selected(:),node_sizes(:)
    logical, allocatable :: keep(:)
    integer :: n,k,i,e,left,right,n_edges,new_node
    real(dp) :: reach
    n=graph%n_rows; k=self%min_samples; if(k==0) k=self%min_cluster_size
    self%core_distances_=spread(infinity(),1,n)
    do i=1,n
      ! Source counts stored entries, including explicit zeros and the diagonal.
      if(graph%indptr(i+1)-graph%indptr(i)>=k) &
        self%core_distances_(i)=kth_value(graph%data(graph%indptr(i):graph%indptr(i+1)-1),k)
    end do
    allocate(rows(graph%nnz()),parent(n),ranks(n),keep(graph%nnz())); parent=[(i,i=1,n)]; ranks=0; keep=.false.
    do i=1,n
      do e=graph%indptr(i),graph%indptr(i+1)-1
        rows(e)=i
        if(ieee_is_nan(self%core_distances_(i)).or.ieee_is_nan(self%core_distances_(graph%indices(e))).or. &
          ieee_is_nan(graph%data(e))) then
          reach=infinity() ! Reciprocal overflow makes all remaining values nonfinite.
        else
          reach=max(self%core_distances_(i),self%core_distances_(graph%indices(e)),graph%data(e))
        end if
        if(ieee_is_finite(reach)) then
          graph%data(e)=reach
        else if(.not.ieee_is_nan(self%max_distance)) then
          if(self%max_distance>0) graph%data(e)=self%max_distance
        end if
        call union_rank(parent,ranks,i,graph%indices(e))
      end do
      call require(graph%indptr(i+1)-graph%indptr(i)>=k,'HDBSCAN sparse graph has fewer than min_samples stored neighbors')
    end do
    left=find_root(parent,1)
    do i=2,n
      call require(find_root(parent,i)==left,'HDBSCAN sparse mutual reachability graph must be connected')
    end do
    ! SciPy 1.15.3 CSR Kruskal uses a stable weight sort, then returns retained
    ! nonzero edges in CSR row order. sklearn subsequently applies NumPy argsort.
    parent=[(i,i=1,n)]; ranks=0; order=stable_edge_order(graph%data); n_edges=0
    do i=1,size(order)
      e=order(i); left=find_root(parent,rows(e)); right=find_root(parent,graph%indices(e))
      if(left==right) cycle
      keep(e)=.true.; call union_rank(parent,ranks,left,right); n_edges=n_edges+1
      if(n_edges==n-1) exit
    end do
    do e=1,size(keep)
      if(ieee_is_nan(graph%data(e))) cycle
      if(graph%data(e)==0) keep(e)=.false.
    end do
    call require(count(keep)==n-1,'HDBSCAN sparse zero-weight MST cannot define a complete hierarchy')
    selected=pack([(i,i=1,size(keep))],keep)
    call require(.not.any(ieee_is_nan(graph%data(selected))),'HDBSCAN sparse MST contains NaN')
    if(allocated(self%mst_edges_)) deallocate(self%mst_edges_)
    allocate(self%mst_edges_(n-1,2)); self%mst_edges_(:,1)=rows(selected); self%mst_edges_(:,2)=graph%indices(selected)
    self%mst_distances_=graph%data(selected); self%has_infinite_edges_=any(.not.ieee_is_finite(self%mst_distances_))
    order=numpy_argsort(self%mst_distances_)
    deallocate(parent,ranks); allocate(parent(2*n-1),node_sizes(2*n-1)); parent=[(i,i=1,2*n-1)]; node_sizes=1
    allocate(children(n-1,2),heights(n-1),sizes(n-1))
    do i=1,n-1
      left=find_root(parent,self%mst_edges_(order(i),1)); right=find_root(parent,self%mst_edges_(order(i),2))
      new_node=n+i; children(i,:)=[left,right]; heights(i)=self%mst_distances_(order(i))
      sizes(i)=node_sizes(left)+node_sizes(right); parent(left)=new_node; parent(right)=new_node; node_sizes(new_node)=sizes(i)
    end do
  end subroutine
  subroutine fit_csr(self,x)
    class(hdbscan), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: graph
    integer, allocatable :: children(:,:),sizes(:),labels(:)
    real(dp), allocatable :: heights(:),probabilities(:)
    real(dp) :: reciprocal
    integer :: n,k,i
    call x%validate(); call controls(self,x%n_rows,x%n_cols)
    call require(self%metric=='precomputed','CSR HDBSCAN currently requires precomputed distances')
    call require(x%n_rows==x%n_cols,'HDBSCAN precomputed distances must be square')
    call require(all(ieee_is_finite(x%data)),'HDBSCAN sparse distances must be finite')
    n=x%n_rows; call require(n>=2,'HDBSCAN requires at least two finite samples')
    k=self%min_samples; if(k==0) k=self%min_cluster_size
    call require(k<=n,'HDBSCAN min_samples exceeds finite sample count')
    graph=sorted_sparse(x,.true.)
    call require(.not.any(ieee_is_nan(graph%data)),'HDBSCAN sparse duplicate sum produced NaN')
    call require(all(graph%data>=0),'HDBSCAN precomputed distances must be nonnegative')
    call sparse_symmetry(graph)
    reciprocal=ieee_quotient(1._dp,self%alpha)
    do i=1,graph%nnz()
      graph%data(i)=ieee_product(graph%data(i),reciprocal)
    end do
    self%n_features_in_=x%n_cols; self%finite_indices_=[(i,i=1,n)]
    call sparse_hierarchy(self,graph,children,heights,sizes)
    call condense(self,children,heights,sizes,n); call select_clusters(self,n,labels,probabilities)
    self%labels_=labels; self%probabilities_=probabilities; self%n_clusters_=max(0,maxval(labels)+1)
    self%children_=children; self%distances_=heights; self%cluster_sizes_=sizes
    if(allocated(self%centroids_)) deallocate(self%centroids_)
    if(allocated(self%medoids_)) deallocate(self%medoids_)
  end subroutine
  function fit_predict_csr(self,x) result(labels)
    class(hdbscan), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    integer, allocatable :: labels(:)
    call self%fit(x); labels=self%labels_
  end function
  subroutine centers(self,x)
    class(hdbscan), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: indices(:)
    real(dp), allocatable :: distances(:,:),strength(:),cost(:)
    integer :: c,i,best
    if(self%store_centers=='centroid'.or.self%store_centers=='both') allocate(self%centroids_(self%n_clusters_,size(x,2)))
    if(self%store_centers=='medoid'.or.self%store_centers=='both') allocate(self%medoids_(self%n_clusters_,size(x,2)))
    do c=0,self%n_clusters_-1
      indices=pack([(i,i=1,size(x,1))],self%labels_==c); strength=self%probabilities_(indices)
      call require(sum(strength)>0,'HDBSCAN cluster center has zero membership strength')
      if(allocated(self%centroids_)) self%centroids_(c+1,:)= &
        matmul(strength,x(indices,:))/sum(strength)
      if(allocated(self%medoids_)) then
        distances=pairwise_distances(x(indices,:),x(indices,:),self%metric,self%p)
        do i=1,size(indices)
          distances(i,i)=0
        end do
        cost=matmul(distances,strength); best=minloc(cost,dim=1); self%medoids_(c+1,:)=x(indices(best),:)
      end if
    end do
  end subroutine
  function fit_predict(self,x) result(labels)
    class(hdbscan), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
  function dbscan_clustering(self,cut_distance,min_cluster_size) result(labels)
    class(hdbscan), intent(in) :: self
    real(dp), intent(in) :: cut_distance
    integer, optional, intent(in) :: min_cluster_size
    integer, allocatable :: labels(:)
    integer, allocatable :: parent(:),ranks(:),sizes(:),mapping(:)
    integer :: n,i,left,right,node,minimum,number
    call require(allocated(self%children_),'HDBSCAN is not fitted')
    call require(.not.ieee_is_nan(cut_distance),'HDBSCAN cut distance must be nonnegative')
    call require(cut_distance>=0,'HDBSCAN cut distance must be nonnegative')
    minimum=5
    if(present(min_cluster_size)) minimum=min_cluster_size
    call require(minimum>=1,'HDBSCAN cut min_cluster_size must be positive')
    n=size(self%labels_); allocate(parent(2*n-1),ranks(2*n-1),sizes(2*n-1),mapping(2*n-1),labels(n))
    parent=[(i,i=1,2*n-1)]; ranks=0; sizes=0; mapping=-1
    do i=1,n-1
      if(self%distances_(i)>=cut_distance) cycle
      call union_rank(parent,ranks,self%children_(i,1),n+i)
      call union_rank(parent,ranks,self%children_(i,2),n+i)
    end do
    do i=1,n
      node=find_root(parent,i); labels(i)=node; sizes(node)=sizes(node)+1
    end do
    number=0
    do i=1,2*n-1
      if(sizes(i)<minimum) cycle
      mapping(i)=number; number=number+1
    end do
    do i=1,n
      labels(i)=mapping(labels(i))
      if(self%labels_(i)<-1) labels(i)=self%labels_(i)
    end do
  end function
end module
