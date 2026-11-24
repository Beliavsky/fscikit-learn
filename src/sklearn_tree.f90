module sklearn_tree
  use sklearn_sparse, only: csr_matrix
  use sklearn_tree_pruning, only: tree_pruning_path,weakest_link_pruning
  use sklearn_tree_median, only: tree_median_queue
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean,ieee_argmax
  use, intrinsic :: iso_fortran_env, only: int64,real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_negative_inf, &
    ieee_positive_inf
  use sklearn_common
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: tree_pruning_path
  public :: decision_tree_classifier,decision_tree_regressor,extra_tree_classifier,extra_tree_regressor
  type :: tree_node
    integer :: left=0,right=0,feature=0,n_samples=0,depth=0
    logical :: missing_go_left=.false.
    real(dp) :: weighted_n_samples=0,threshold=0,impurity=0,value=0
    real(dp), allocatable :: probability(:)
  end type
  type :: tree_storage
    integer :: max_depth=100,min_samples_split=2,min_samples_leaf=1,max_features=0
    integer :: random_state=0,node_count_=0,n_features_in_=0
    logical :: random_thresholds=.false.
    logical, private :: weighted_fit_=.false.
    integer :: max_features_=0
    real(dp) :: min_impurity_decrease=0,ccp_alpha=0,min_weight_fraction_leaf=0
    real(dp), allocatable :: feature_importances_(:)
    type(tree_node), allocatable :: nodes_(:)
  contains
    procedure :: decision_path=>tree_decision_path
    procedure :: apply=>tree_apply
    procedure :: get_depth=>tree_depth
    procedure :: get_n_leaves=>tree_leaves
  end type
  type, extends(tree_storage) :: decision_tree_classifier
    character(16) :: criterion='gini'
    integer, allocatable :: classes_(:)
  contains
    procedure :: cost_complexity_pruning_path=>classifier_pruning_path
    procedure :: fit=>classifier_fit
    procedure :: predict_proba=>classifier_proba
    procedure :: predict=>classifier_predict
    procedure :: score=>classifier_score
  end type
  type, extends(tree_storage) :: decision_tree_regressor
    character(16) :: criterion='squared_error'
  contains
    procedure :: cost_complexity_pruning_path=>regressor_pruning_path
    procedure :: fit=>regressor_fit
    procedure :: predict=>regressor_predict
    procedure :: score=>regressor_score
  end type
  type, extends(decision_tree_classifier) :: extra_tree_classifier
  contains
    procedure :: fit=>extra_classifier_fit
  end type
  type, extends(decision_tree_regressor) :: extra_tree_regressor
  contains
    procedure :: fit=>extra_regressor_fit
  end type

contains
  recursive real(dp) function tree_numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8),left,right
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=ieee_difference(partial(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
      right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(tree_numpy_sum(a(:split)),-tree_numpy_sum(a(split+1:)))
    end if
  end function
  ! Weighted/randomized splitter follows sklearn 1.6.1 tree/_splitter.pyx and _criterion.pyx.
  subroutine build_weighted_tree(self,x,y,classes,criterion,w,max_features)
    class(tree_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),w(:)
    integer, intent(in) :: classes(:)
    character(*), intent(in) :: criterion
    integer, intent(in), optional :: max_features
    type(legacy_random_state) :: rng
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: active(:)
    integer :: all_features(size(x,2)),constants(size(x,2)),i,root,nf
    integer(int64) :: state
    real(dp) :: root_weight,min_weight,normalizer,contribution
    type(tree_median_queue) :: median_left,median_right
    logical :: source_regression,work_splitter
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    if(size(classes)==0) then
      call require(any(criterion==[character(16)::'squared_error','friedman_mse','poisson','absolute_error']),'unknown regression tree criterion')
      if(criterion=='poisson') then
        call require(all(y>=0),'Poisson tree targets must be nonnegative')
        call require(ieee_total(y)>0,'Poisson tree target sum must be positive')
      end if
    end if
    source_regression=size(classes)==0
    work_splitter=size(classes)>0.or.source_regression
    call require(size(w)==size(y),'tree sample weight length mismatch')
    call require(all(ieee_is_finite(w)),'tree sample weights must be finite')
    call require(size(y)<=huge(1)/2,'tree node capacity exceeds integer range')
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'tree training features exceed float32 range')
    call require(self%max_depth>=0.and.self%min_samples_split>=2.and.self%min_samples_leaf>=1, &
      'invalid tree size controls')
    call require(self%max_features>=0.and.self%max_features<=size(x,2),'invalid tree max_features')
    call require(ieee_is_finite(self%min_impurity_decrease),'tree impurity decrease must be finite')
    call require(self%min_impurity_decrease>=0,'negative impurity decrease')
    call require(ieee_is_finite(self%min_weight_fraction_leaf),'tree minimum weighted leaf fraction must be finite')
    call require(self%min_weight_fraction_leaf>=0.and.self%min_weight_fraction_leaf<=.5_dp, &
      'tree minimum weighted leaf fraction must be in [0,0.5]')
    call require(ieee_is_finite(self%ccp_alpha),'tree ccp_alpha must be finite')
    call require(self%ccp_alpha>=0,'tree ccp_alpha must be nonnegative')
    call require(self%random_state>=0,'weighted tree random_state must be nonnegative')
    nf=self%max_features; if(nf==0) nf=size(x,2)
    if(present(max_features)) nf=max_features
    self%max_features_=nf; self%weighted_fit_=.true.; self%n_features_in_=size(x,2); self%node_count_=0
    if(allocated(self%nodes_)) deallocate(self%nodes_)
    allocate(self%nodes_(2*size(y))); self%feature_importances_=spread(0._dp,1,size(x,2))
    z=real(real(x,real32),dp); active=pack([(i,i=1,size(y))],w/=0)
    root_weight=ieee_total(w); min_weight=ieee_product(self%min_weight_fraction_leaf,tree_numpy_sum(w))
    all_features=[(i,i=1,size(x,2))]; constants=0
    call rng%seed(self%random_state); state=int(rng%integer_index(2147483647)-1,int64)
    ! MAE keeps its initialized queue storage across every node reset.
    if(criterion=='absolute_error') then
      call median_left%init(size(y));call median_right%init(size(y))
    end if
    root=grow(active,0,0,0._dp)
    if(self%ccp_alpha/=0) call prune_fitted_tree(self)
    do i=1,self%node_count_
      associate(node=>self%nodes_(i))
        if(node%left==0) cycle
        contribution=ieee_product(node%weighted_n_samples,node%impurity)
        contribution=ieee_difference(contribution, &
          ieee_product(self%nodes_(node%left)%weighted_n_samples,self%nodes_(node%left)%impurity))
        contribution=ieee_difference(contribution, &
          ieee_product(self%nodes_(node%right)%weighted_n_samples,self%nodes_(node%right)%impurity))
        self%feature_importances_(node%feature)=ieee_difference(self%feature_importances_(node%feature),-contribution)
      end associate
    end do
    do i=1,size(self%feature_importances_)
      self%feature_importances_(i)=ieee_quotient(self%feature_importances_(i),root_weight)
    end do
    normalizer=ieee_total(self%feature_importances_)
    if(.not.ieee_is_nan(normalizer)) then
      if(normalizer>0) then
        do i=1,size(self%feature_importances_)
          self%feature_importances_(i)=ieee_quotient(self%feature_importances_(i),normalizer)
        end do
      end if
    end if
  contains
    integer function next_random() result(value)
      if(state==0) state=1
      state=iand(ieor(state,ishft(state,13)),4294967295_int64)
      state=ieor(state,ishft(state,-17))
      state=iand(ieor(state,ishft(state,5)),4294967295_int64)
      value=int(iand(state,2147483647_int64))
    end function
    subroutine swap(a,b)
      integer, intent(inout) :: a,b
      integer :: temporary
      temporary=a; a=b; b=temporary
    end subroutine
    logical function less(a,b) result(value)
      real(dp), intent(in) :: a,b
      value=.false.
      if(ieee_is_nan(a).or.ieee_is_nan(b)) return
      value=a<b
    end function
    subroutine moments(ids,mass,sums,squares)
      integer, intent(in) :: ids(:)
      real(dp), intent(out) :: mass,sums(:),squares
      real(dp) :: wy
      integer :: a,c,idx
      mass=0; sums=0; squares=0
      do a=1,size(ids)
        idx=ids(a); mass=ieee_difference(mass,-w(idx))
        if(size(classes)==0) then
          wy=ieee_product(w(idx),y(idx)); sums(1)=ieee_difference(sums(1),-wy)
          squares=ieee_difference(squares,-ieee_product(wy,y(idx)))
        else
          do c=1,size(classes)
            if(y(idx)==real(classes(c),dp)) sums(c)=ieee_difference(sums(c),-w(idx))
          end do
        end if
      end do
    end subroutine
    real(dp) function tree_log(x) result(value)
      real(dp), intent(in) :: x
      value=ieee_value(0._dp,ieee_quiet_nan)
      if(ieee_is_nan(x)) return
      if(x==0) then
        value=ieee_value(0._dp,ieee_negative_inf)
      else if(x>0) then
        value=log(x)
      end if
    end function
    real(dp) function regression_squares(ids) result(value)
      integer, intent(in) :: ids(:)
      integer :: i,index_
      value=0
      do i=1,size(ids)
        index_=ids(i)
        value=ieee_difference(value,-ieee_product(ieee_product(w(index_),y(index_)),y(index_)))
      end do
    end function
    real(dp) function impurity(mass,sums,squares,ids) result(value)
      real(dp), intent(in) :: mass,sums(:),squares
      integer, optional, intent(in) :: ids(:)
      real(dp) :: probability,term,total,ratio
      integer :: c,i,index_
      if(size(classes)==0) then
        if(criterion=='poisson') then
          value=ieee_value(0._dp,ieee_positive_inf)
          if(.not.ieee_is_nan(sums(1))) then
            if(sums(1)<=10*epsilon(1._dp)) return
          end if
          probability=ieee_quotient(sums(1),mass);total=0
          do i=1,size(ids)
            index_=ids(i);ratio=ieee_quotient(y(index_),probability)
            term=0
            if(ieee_is_nan(ratio).or.y(index_)/=0) term=ieee_product(y(index_),tree_log(ratio))
            total=ieee_difference(total,-ieee_product(w(index_),term))
          end do
          value=ieee_quotient(total,mass);return
        end if
        probability=ieee_quotient(sums(1),mass)
        value=ieee_difference(ieee_quotient(squares,mass),ieee_product(probability,probability))
      else if(criterion=='gini') then
        total=0
        do c=1,size(classes)
          total=ieee_difference(total,-ieee_product(sums(c),sums(c)))
        end do
        value=ieee_difference(1._dp,ieee_quotient(total,ieee_product(mass,mass)))
      else
        value=0
        do c=1,size(classes)
          if(ieee_is_nan(sums(c))) cycle
          if(sums(c)<=0) cycle
          probability=ieee_quotient(sums(c),mass)
          term=ieee_value(0._dp,ieee_quiet_nan)
          if(.not.ieee_is_nan(probability)) then
            if(probability>0) term=ieee_product(probability,log(probability)/log(2._dp))
          end if
          value=ieee_difference(value,term)
        end do
      end if
    end function
    subroutine sort_swap(values,ids,a,b)
      real(dp), intent(inout) :: values(:)
      integer, intent(inout) :: ids(:)
      integer, intent(in) :: a,b
      real(dp) :: value
      integer :: index
      value=values(a);values(a)=values(b);values(b)=value
      index=ids(a);ids(a)=ids(b);ids(b)=index
    end subroutine
    subroutine sift(values,ids,base,start,finish)
      real(dp), intent(inout) :: values(:)
      integer, intent(inout) :: ids(:)
      integer, intent(in) :: base,start,finish
      integer :: root,child,largest
      root=start
      do
        child=2*root+1;largest=root
        if(child<finish) then
          if(values(base+largest)<values(base+child)) largest=child
        end if
        if(child+1<finish) then
          if(values(base+largest)<values(base+child+1)) largest=child+1
        end if
        if(largest==root) exit
        call sort_swap(values,ids,base+root,base+largest);root=largest
      end do
    end subroutine
    subroutine heap_sort(values,ids,base,n)
      real(dp), intent(inout) :: values(:)
      integer, intent(inout) :: ids(:)
      integer, intent(in) :: base,n
      integer :: start,finish
      start=(n-2)/2
      do
        call sift(values,ids,base,start,n)
        if(start==0) exit
        start=start-1
      end do
      finish=n-1
      do while(finish>0)
        call sort_swap(values,ids,base,base+finish)
        call sift(values,ids,base,0,finish);finish=finish-1
      end do
    end subroutine
    recursive subroutine intro_sort(values,ids,first,length,maximum_depth)
      real(dp), intent(inout) :: values(:)
      integer, intent(inout) :: ids(:)
      integer, intent(in) :: first,length,maximum_depth
      real(dp) :: a,b,c,pivot
      integer :: n,base,depth,i,left,right
      n=length;base=first;depth=maximum_depth
      do while(n>1)
        if(depth<=0) then
          call heap_sort(values,ids,base,n);return
        end if
        depth=depth-1;a=values(base);b=values(base+n/2);c=values(base+n-1)
        if(a<b) then
          if(b<c) then
            pivot=b
          else if(a<c) then
            pivot=c
          else
            pivot=a
          end if
        else if(b<c) then
          if(a<c) then
            pivot=a
          else
            pivot=c
          end if
        else
          pivot=b
        end if
        i=0;left=0;right=n
        do while(i<right)
          if(values(base+i)<pivot) then
            call sort_swap(values,ids,base+i,base+left);i=i+1;left=left+1
          else if(values(base+i)>pivot) then
            right=right-1;call sort_swap(values,ids,base+i,base+right)
          else
            i=i+1
          end if
        end do
        call intro_sort(values,ids,base,left,depth);base=base+right;n=n-right
      end do
    end subroutine
    subroutine sample_sort(values,ids)
      real(dp), intent(inout) :: values(:)
      integer, intent(inout) :: ids(:)
      integer :: remaining,depth
      remaining=size(ids);depth=0
      do while(remaining>1)
        remaining=remaining/2;depth=depth+1
      end do
      call intro_sort(values,ids,1,size(ids),2*depth)
    end subroutine
    subroutine classification_update(ids,cut,pos,mass,sums,wl,sl,wr,sr)
      integer, intent(in) :: ids(:),cut
      integer, intent(inout) :: pos
      real(dp), intent(in) :: mass,sums(:)
      real(dp), intent(inout) :: wl,sl(:)
      real(dp), intent(out) :: wr,sr(:)
      integer :: j,c,index
      if(cut-pos<=size(ids)-cut) then
        do j=pos+1,cut
          index=ids(j)
          if(size(classes)==0) then
            sl(1)=ieee_difference(sl(1),-ieee_product(w(index),y(index)))
          else
            c=label_index(classes,int(y(index)));sl(c)=ieee_difference(sl(c),-w(index))
          end if
          wl=ieee_difference(wl,-w(index))
        end do
      else
        sl=sums;wl=mass
        do j=size(ids),cut+1,-1
          index=ids(j)
          if(size(classes)==0) then
            sl(1)=ieee_difference(sl(1),ieee_product(w(index),y(index)))
          else
            c=label_index(classes,int(y(index)));sl(c)=ieee_difference(sl(c),w(index))
          end if
          wl=ieee_difference(wl,w(index))
        end do
      end if
      wr=ieee_difference(mass,wl)
      do c=1,size(sums)
        sr(c)=ieee_difference(sums(c),sl(c))
      end do
      pos=cut
    end subroutine
    subroutine median_reset(left,right,reverse)
      type(tree_median_queue), intent(inout) :: left,right
      logical, intent(in) :: reverse
      real(dp) :: value,weight
      if(reverse) then
        do while(right%count()>0)
          call right%pop(value,weight);call left%push(value,weight)
        end do
      else
        do while(left%count()>0)
          call left%pop(value,weight);call right%push(value,weight)
        end do
      end if
    end subroutine
    subroutine median_update(ids,cut,pos,mass,wl,wr,left,right)
      integer, intent(in) :: ids(:),cut
      integer, intent(inout) :: pos
      real(dp), intent(in) :: mass
      real(dp), intent(inout) :: wl
      real(dp), intent(out) :: wr
      type(tree_median_queue), intent(inout) :: left,right
      integer :: j,index_
      if(cut-pos<=size(ids)-cut) then
        do j=pos+1,cut
          index_=ids(j)
          call right%remove(y(index_),w(index_));call left%push(y(index_),w(index_))
          wl=ieee_difference(wl,-w(index_))
        end do
      else
        call median_reset(left,right,.true.);wl=mass
        do j=size(ids),cut+1,-1
          index_=ids(j)
          call left%remove(y(index_),w(index_));call right%push(y(index_),w(index_))
          wl=ieee_difference(wl,w(index_))
        end do
      end if
      wr=ieee_difference(mass,wl);pos=cut
    end subroutine
    real(dp) function absolute_impurity(ids,center,mass) result(value)
      integer, intent(in) :: ids(:)
      real(dp), intent(in) :: center,mass
      integer :: j,index_
      value=0
      do j=1,size(ids)
        index_=ids(j)
        value=ieee_difference(value,-ieee_product(abs(ieee_difference(y(index_),center)),w(index_)))
      end do
      value=ieee_quotient(value,mass)
    end function
    subroutine partition_final(ids,feature,threshold)
      integer, intent(inout) :: ids(:)
      integer, intent(in) :: feature
      real(dp), intent(in) :: threshold
      integer :: at,last
      at=1;last=size(ids)
      do while(at<last)
        if(z(ids(at),feature)<=threshold) then
          at=at+1
        else
          call swap(ids(at),ids(last));last=last-1
        end if
      end do
    end subroutine
    recursive integer function grow(ids,depth,known,inherited) result(node)
      integer, intent(in) :: ids(:),depth,known
      real(dp), intent(in) :: inherited
      real(dp) :: node_median
      integer, allocatable :: ordered(:),left(:),right(:),work(:)
      real(dp), allocatable :: feature_values(:)
      real(dp) :: sums(max(1,size(classes))),sl(max(1,size(classes))),sr(max(1,size(classes)))
      real(dp) :: mass,squares,parent,wl,wr,sql,sqr,il,ir,proxy,best,low,high,threshold,best_threshold
      real(dp) :: best_left,best_right,best_wl,best_wr,improvement,difference
      integer :: n,fi,fj,drawn,found,visited,nconstant,f,cut,j,c,best_feature,left_node,right_node,first,last
      integer :: pos,best_pos,at,partition_end
      n=size(ids);call moments(ids,mass,sums,squares)
      node_median=0
      if(criterion=='absolute_error') then
        call median_left%clear();call median_right%clear()
        do j=1,n
          call median_right%push(y(ids(j)),w(ids(j)))
        end do
        node_median=median_right%median()
      end if
      if(work_splitter) then
        work=ids;allocate(feature_values(n))
      end if
      self%node_count_=self%node_count_+1; node=self%node_count_
      self%nodes_(node)%n_samples=n; self%nodes_(node)%weighted_n_samples=mass; self%nodes_(node)%depth=depth
      parent=inherited
      if(depth==0) then
        if(criterion=='absolute_error') then
          parent=absolute_impurity(ids,node_median,mass)
        else
          parent=impurity(mass,sums,squares,ids)
        end if
      end if
      self%nodes_(node)%impurity=parent
      if(size(classes)==0) then
        if(criterion=='absolute_error') then
          self%nodes_(node)%value=node_median
        else
          self%nodes_(node)%value=ieee_quotient(sums(1),mass)
        end if
      else
        allocate(self%nodes_(node)%probability(size(classes)))
        do c=1,size(classes)
          self%nodes_(node)%probability(c)=ieee_quotient(sums(c),mass)
        end do
      end if
      if(depth>=self%max_depth.or.n<self%min_samples_split.or.n/2<self%min_samples_leaf) return
      if(less(mass,ieee_product(2._dp,min_weight))) return
      if(.not.ieee_is_nan(parent)) then
        if(parent<=epsilon(1._dp)) return
      end if
      fi=size(x,2); found=0; drawn=0; visited=0; nconstant=known
      best=ieee_value(0._dp,ieee_negative_inf); best_feature=0; best_threshold=0
      best_left=0;best_right=0;best_wl=0;best_wr=0;best_pos=0;f=0
      do while(fi>nconstant.and.(visited<nf.or.visited<=found+drawn))
        visited=visited+1; fj=drawn+modulo(next_random(),fi-found-drawn)
        if(fj<known) then
          call swap(all_features(drawn+1),all_features(fj+1)); drawn=drawn+1; cycle
        end if
        fj=fj+found; f=all_features(fj+1)
        if(work_splitter) then
          feature_values=z(work,f)
          if(.not.self%random_thresholds) call sample_sort(feature_values,work)
          ordered=work;low=minval(feature_values);high=maxval(feature_values)
        else
          ordered=ids(argsort(z(ids,f)));low=z(ordered(1),f);high=z(ordered(n),f)
        end if
        ! The tested sklearn 1.6.1 binary exports FEATURE_THRESHOLD=0.0,
        ! despite the 1e-7 initializer in its .pxd source. Keep distinct
        ! float32 values distinct, matching the observed binary behavior.
        if(high<=low) then
          call swap(all_features(fj+1),all_features(nconstant+1)); found=found+1; nconstant=nconstant+1; cycle
        end if
        fi=fi-1; call swap(all_features(fi+1),all_features(fj+1))
        first=self%min_samples_leaf; last=n-self%min_samples_leaf
        if(self%random_thresholds) then
          threshold=(high-low)*real(next_random(),dp)/2147483647._dp+low
          if(threshold==high) threshold=low
          if(work_splitter) then
            at=1;partition_end=n+1
            do while(at<partition_end)
              if(feature_values(at)<=threshold) then
                at=at+1
              else
                partition_end=partition_end-1;call sort_swap(feature_values,work,at,partition_end)
              end if
            end do
            cut=partition_end-1;ordered=work
          else
            cut=count(z(ordered,f)<=threshold)
          end if
          first=cut;last=cut
        end if
        pos=0;wl=0;sl=0
        if(criterion=='absolute_error') call median_reset(median_left,median_right,.false.)
        do cut=first,last
          if(cut<self%min_samples_leaf.or.n-cut<self%min_samples_leaf) cycle
          if(.not.self%random_thresholds) then
            if(z(ordered(cut+1),f)<=z(ordered(cut),f)) cycle
            threshold=z(ordered(cut),f)/2+z(ordered(cut+1),f)/2
            if(threshold==z(ordered(cut+1),f)) threshold=z(ordered(cut),f)
          end if
          if(size(classes)>0) then
            call classification_update(ordered,cut,pos,mass,sums,wl,sl,wr,sr);sql=0;sqr=0
          else if(criterion=='absolute_error') then
            call median_update(ordered,cut,pos,mass,wl,wr,median_left,median_right);sql=0;sqr=0
          else if(source_regression) then
            call classification_update(ordered,cut,pos,mass,sums,wl,sl,wr,sr)
            sql=regression_squares(ordered(:cut));sqr=ieee_difference(squares,sql)
          else
            call moments(ordered(:cut),wl,sl,sql);wr=ieee_difference(mass,wl)
            do c=1,size(sums)
              sr(c)=ieee_difference(sums(c),sl(c))
            end do
            sqr=ieee_difference(squares,sql)
          end if
          if(less(wl,min_weight).or.less(wr,min_weight)) cycle
          if(criterion=='absolute_error') then
            il=absolute_impurity(ordered(:cut),median_left%median(),wl)
            ir=absolute_impurity(ordered(cut+1:),median_right%median(),wr)
          else
            il=impurity(wl,sl,sql,ordered(:cut));ir=impurity(wr,sr,sqr,ordered(cut+1:))
          end if
          if(size(classes)==0) then
            select case(criterion)
            case('absolute_error')
              proxy=ieee_difference(-ieee_product(wr,ir),ieee_product(wl,il))
            case('friedman_mse')
              difference=ieee_difference(ieee_product(wr,sl(1)),ieee_product(wl,sr(1)))
              proxy=ieee_quotient(ieee_product(difference,difference),ieee_product(wl,wr))
            case('poisson')
              ! sklearn.tree._utils.log uses base two for the split proxy.
              ! The deviance itself uses the natural logarithm (scipy xlogy).
              if(.not.ieee_is_nan(sl(1))) then
                if(sl(1)<=10*epsilon(1._dp)) cycle
              end if
              if(.not.ieee_is_nan(sr(1))) then
                if(sr(1)<=10*epsilon(1._dp)) cycle
              end if
              proxy=ieee_difference(ieee_product(sl(1),ieee_quotient(tree_log(ieee_quotient(sl(1),wl)),log(2._dp))), &
                -ieee_product(sr(1),ieee_quotient(tree_log(ieee_quotient(sr(1),wr)),log(2._dp))))
            case default
              proxy=ieee_difference(ieee_quotient(ieee_product(sl(1),sl(1)),wl), &
                -ieee_quotient(ieee_product(sr(1),sr(1)),wr))
            end select
          else
            proxy=ieee_difference(-ieee_product(wr,ir),ieee_product(wl,il))
          end if
          if(ieee_is_nan(proxy)) cycle
          if(proxy>best) then
            best=proxy;best_feature=f;best_threshold=threshold;best_pos=cut
            best_left=il; best_right=ir; best_wl=wl; best_wr=wr
          end if
        end do
      end do
      all_features(:known)=constants(:known); constants(known+1:nconstant)=all_features(known+1:nconstant)
      if(best_feature==0) return
      if(work_splitter) then
        if(.not.self%random_thresholds.or.f/=best_feature) call partition_final(work,best_feature,best_threshold)
        pos=0;wl=0;sl=0
        if(criterion=='absolute_error') then
          call median_reset(median_left,median_right,.false.)
          call median_update(work,best_pos,pos,mass,wl,wr,median_left,median_right)
          best_left=absolute_impurity(work(:best_pos),median_left%median(),wl)
          best_right=absolute_impurity(work(best_pos+1:),median_right%median(),wr)
        else
          call classification_update(work,best_pos,pos,mass,sums,wl,sl,wr,sr)
          sql=0;sqr=0
          if(source_regression) then
            sql=regression_squares(work(:best_pos));sqr=ieee_difference(squares,sql)
          end if
          best_left=impurity(wl,sl,sql,work(:best_pos))
          best_right=impurity(wr,sr,sqr,work(best_pos+1:))
        end if
        best_wl=wl;best_wr=wr
      end if
      improvement=ieee_difference(parent,ieee_product(ieee_quotient(best_wr,mass),best_right))
      improvement=ieee_difference(improvement,ieee_product(ieee_quotient(best_wl,mass),best_left))
      improvement=ieee_product(ieee_quotient(mass,root_weight),improvement)
      if(criterion=='friedman_mse') then
        difference=ieee_difference(ieee_product(wr,sl(1)),ieee_product(wl,sr(1)))
        improvement=ieee_quotient(ieee_product(difference,difference),ieee_product(ieee_product(wl,wr),mass))
      end if
      if(.not.ieee_is_nan(improvement)) then
        if(ieee_difference(improvement,-epsilon(1._dp))<self%min_impurity_decrease) return
      end if
      if(work_splitter) then
        left=work(:best_pos);right=work(best_pos+1:)
      else
        left=pack(ids,z(ids,best_feature)<=best_threshold);right=pack(ids,z(ids,best_feature)>best_threshold)
      end if
      self%nodes_(node)%feature=best_feature; self%nodes_(node)%threshold=best_threshold
      self%nodes_(node)%missing_go_left=size(left)>size(right)
      left_node=grow(left,depth+1,nconstant,best_left); right_node=grow(right,depth+1,nconstant,best_right)
      self%nodes_(node)%left=left_node; self%nodes_(node)%right=right_node
    end function
  end subroutine

  subroutine classifier_fit(self,x,y,sample_weight)
    class(decision_tree_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call check_xy(x,size(y))
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'unknown tree criterion')
    self%classes_=unique_labels(y)
    if(present(sample_weight)) then
      call build_weighted_tree(self,x,real(y,dp),self%classes_,self%criterion,sample_weight)
    else
      call build_weighted_tree(self,x,real(y,dp),self%classes_,self%criterion,spread(1._dp,1,size(y)))
    end if
  end subroutine
  subroutine regressor_fit(self,x,y,sample_weight)
    class(decision_tree_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    if(present(sample_weight)) then
      call build_weighted_tree(self,x,y,[integer::],self%criterion,sample_weight)
    else
      call build_weighted_tree(self,x,y,[integer::],self%criterion,spread(1._dp,1,size(y)))
    end if
  end subroutine
  subroutine extra_classifier_fit(self,x,y,sample_weight)
    class(extra_tree_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(size(y))
    integer :: nf
    call check_xy(x,size(y)); self%random_thresholds=.true.
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'unknown tree criterion')
    self%classes_=unique_labels(y); weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'tree sample weight length mismatch'); weights=sample_weight
    end if
    nf=self%max_features; if(nf==0) nf=max(1,int(sqrt(real(size(x,2),dp))))
    call build_weighted_tree(self,x,real(y,dp),self%classes_,self%criterion,weights,nf)
  end subroutine
  subroutine extra_regressor_fit(self,x,y,sample_weight)
    class(extra_tree_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(size(y))
    call check_xy(x,size(y)); self%random_thresholds=.true.; weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'tree sample weight length mismatch'); weights=sample_weight
    end if
    call build_weighted_tree(self,x,y,[integer::],self%criterion,weights)
  end subroutine
  subroutine pruning_mask(self,leaves,path,alpha)
    class(tree_storage), intent(in) :: self
    logical, allocatable, intent(out) :: leaves(:)
    type(tree_pruning_path), intent(out) :: path
    real(dp), optional, intent(in) :: alpha
    integer :: left(self%node_count_),right(self%node_count_),i
    real(dp) :: mass(self%node_count_),impurity(self%node_count_)
    do i=1,self%node_count_
      left(i)=self%nodes_(i)%left;right(i)=self%nodes_(i)%right
      mass(i)=self%nodes_(i)%weighted_n_samples;impurity(i)=self%nodes_(i)%impurity
    end do
    call weakest_link_pruning(left,right,mass,impurity,leaves,path,alpha)
  end subroutine
  function fitted_pruning_path(self) result(path)
    class(tree_storage), intent(in) :: self
    type(tree_pruning_path) :: path
    logical, allocatable :: leaves(:)
    call pruning_mask(self,leaves,path)
  end function
  subroutine prune_fitted_tree(self)
    class(tree_storage), intent(inout) :: self
    type(tree_pruning_path) :: path
    type(tree_node), allocatable :: original(:)
    logical, allocatable :: leaves(:)
    integer :: root
    call pruning_mask(self,leaves,path,self%ccp_alpha)
    call move_alloc(self%nodes_,original)
    allocate(self%nodes_(2*count(leaves)-1));self%node_count_=0
    root=copy_branch(1,0)
  contains
    recursive integer function copy_branch(old,depth) result(node)
      integer, intent(in) :: old,depth
      integer :: left,right
      self%node_count_=self%node_count_+1;node=self%node_count_
      self%nodes_(node)=original(old);self%nodes_(node)%depth=depth
      if(leaves(old)) then
        self%nodes_(node)%left=0;self%nodes_(node)%right=0
        self%nodes_(node)%feature=0;self%nodes_(node)%threshold=0
      else
        left=copy_branch(original(old)%left,depth+1);right=copy_branch(original(old)%right,depth+1)
        self%nodes_(node)%left=left;self%nodes_(node)%right=right
      end if
    end function
  end subroutine
  function classifier_pruning_path(self,x,y,sample_weight) result(path)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(tree_pruning_path) :: path
    class(decision_tree_classifier), allocatable :: copied
    allocate(copied,source=self);copied%ccp_alpha=0
    call copied%fit(x,y,sample_weight);path=fitted_pruning_path(copied)
  end function
  function regressor_pruning_path(self,x,y,sample_weight) result(path)
    class(decision_tree_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(tree_pruning_path) :: path
    class(decision_tree_regressor), allocatable :: copied
    allocate(copied,source=self);copied%ccp_alpha=0
    call copied%fit(x,y,sample_weight);path=fitted_pruning_path(copied)
  end function
  subroutine check_tree_queries(self,x)
    class(tree_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: i,j
    call require(allocated(self%nodes_),'decision tree is not fitted')
    if(self%weighted_fit_) then
      call require(size(x,1)>0.and.size(x,2)==self%n_features_in_,'tree query feature shape mismatch')
      select type(self)
      class is(decision_tree_regressor)
        if(self%criterion=='absolute_error') call require(all(ieee_is_finite(x)),'absolute-error trees reject missing queries')
      end select
      call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'tree queries contain infinity')
      do j=1,size(x,2)
        do i=1,size(x,1)
          if(ieee_is_nan(x(i,j))) cycle
          call require(abs(x(i,j))<=real(huge(0._real32),dp),'tree query features exceed float32 range')
        end do
      end do
    else
      call check_xy(transpose(x),self%n_features_in_)
    end if
  end subroutine
  function tree_apply(self,x) result(leaves)
    class(tree_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: leaves(size(x,1)),i,node,f,j
    real(dp) :: value
    call check_tree_queries(self,x)
    do i=1,size(x,1)
      node=1
      do while(self%nodes_(node)%left>0)
        f=self%nodes_(node)%feature
        if(self%weighted_fit_) then
          if(ieee_is_nan(x(i,f))) then
            if(self%nodes_(node)%missing_go_left) then
              node=self%nodes_(node)%left
            else
              node=self%nodes_(node)%right
            end if
            cycle
          end if
          value=real(real(x(i,f),real32),dp)
        else
          value=x(i,f)
        end if
        if(value<=self%nodes_(node)%threshold) then
          node=self%nodes_(node)%left
        else
          node=self%nodes_(node)%right
        end if
      end do
      leaves(i)=node
    end do
  end function
  integer function indicator_child(self,node,row) result(child)
    class(tree_storage), intent(in) :: self
    integer, intent(in) :: node
    real(dp), intent(in) :: row(:)
    integer :: f
    real(dp) :: value
    f=self%nodes_(node)%feature
    ! sklearn 1.6.1 _decision_path_dense sends NaNs right, unlike apply.
    child=self%nodes_(node)%right
    if(ieee_is_nan(row(f))) return
    value=row(f)
    if(self%weighted_fit_) value=real(real(value,real32),dp)
    if(value<=self%nodes_(node)%threshold) child=self%nodes_(node)%left
  end function
  function tree_decision_path(self,x) result(indicator)
    class(tree_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix) :: indicator
    integer :: i,node,k
    integer(int64) :: stored
    call check_tree_queries(self,x)
    call require(size(x,1)<huge(1),'tree decision-path row count exceeds integer range')
    indicator%n_rows=size(x,1);indicator%n_cols=self%node_count_
    allocate(indicator%indptr(size(x,1)+1));indicator%indptr(1)=1;stored=1
    do i=1,size(x,1)
      node=1
      do
        stored=stored+1
        call require(stored<=int(huge(1),int64),'tree decision-path stored count exceeds integer range')
        if(self%nodes_(node)%left==0) exit
        node=indicator_child(self,node,x(i,:))
      end do
      indicator%indptr(i+1)=int(stored)
    end do
    allocate(indicator%indices(int(stored)-1),indicator%data(int(stored)-1));indicator%data=1
    do i=1,size(x,1)
      node=1;k=indicator%indptr(i)
      do
        indicator%indices(k)=node;k=k+1
        if(self%nodes_(node)%left==0) exit
        node=indicator_child(self,node,x(i,:))
      end do
    end do
    call indicator%validate()
  end function
  integer function tree_depth(self) result(depth)
    class(tree_storage), intent(in) :: self
    call require(allocated(self%nodes_),'decision tree is not fitted')
    depth=maxval(self%nodes_(:self%node_count_)%depth)
  end function
  integer function tree_leaves(self) result(n)
    class(tree_storage), intent(in) :: self
    call require(allocated(self%nodes_),'decision tree is not fitted')
    n=count(self%nodes_(:self%node_count_)%left==0)
  end function
  function classifier_proba(self,x) result(prob)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: leaves(size(x,1)),i
    leaves=self%apply(x); allocate(prob(size(x,1),size(self%classes_)))
    do i=1,size(x,1)
      prob(i,:)=self%nodes_(leaves(i))%probability
    end do
  end function
  function classifier_predict(self,x) result(y)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      if(self%weighted_fit_) then
        y(i)=self%classes_(ieee_argmax(prob(i,:)))
      else
        y(i)=self%classes_(maxloc(prob(i,:),dim=1))
      end if
    end do
  end function
  real(dp) function classifier_score(self,x,y,sample_weight) result(v)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    real(dp) :: mass
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'tree score target length mismatch')
    call check_score_weights(sample_weight,size(y))
    if(present(sample_weight)) then
      mass=ieee_total(sample_weight)
      if(.not.ieee_is_nan(mass)) call require(mass/=0,'tree score weights must have a nonzero sum')
      v=ieee_mean(merge(1._dp,0._dp,y==predicted),sample_weight)
    else
      v=accuracy_score(y,predicted)
    end if
  end function
  function regressor_predict(self,x) result(y)
    class(decision_tree_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    integer :: leaves(size(x,1)),i
    leaves=self%apply(x)
    do i=1,size(y)
      y(i)=self%nodes_(leaves(i))%value
    end do
  end function
  real(dp) function regressor_score(self,x,y,sample_weight) result(v)
    class(decision_tree_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: predicted(:),numerators(:),denominators(:)
    real(dp) :: center,num,den,mass
    integer :: i
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'tree score target length mismatch')
    call require(all(ieee_is_finite(y)),'tree regression score targets must be finite')
    call require(all(ieee_is_finite(predicted)),'tree regression score predictions must be finite')
    call check_score_weights(sample_weight,size(y))
    if(size(y)<2) then
      v=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    if(present(sample_weight)) then
      mass=ieee_total(sample_weight)
      if(.not.ieee_is_nan(mass)) call require(mass/=0,'tree score weights must have a nonzero sum')
    end if
    center=ieee_mean(y,sample_weight); allocate(numerators(size(y)),denominators(size(y)))
    do i=1,size(y)
      num=ieee_difference(y(i),predicted(i)); numerators(i)=ieee_product(num,num)
      den=ieee_difference(y(i),center); denominators(i)=ieee_product(den,den)
    end do
    num=ieee_mean(numerators,sample_weight); den=ieee_mean(denominators,sample_weight)
    v=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(num).or.ieee_is_nan(den)) return
    v=1
    if(num/=0) then
      v=0
      if(den/=0) v=ieee_difference(1._dp,ieee_quotient(num,den))
    end if
  end function
  subroutine check_score_weights(weights,n)
    real(dp), intent(in), optional :: weights(:)
    integer, intent(in) :: n
    if(.not.present(weights)) return
    call require(size(weights)==n,'tree score weight length mismatch')
    call require(all(ieee_is_finite(weights)),'tree score weights must be finite')
  end subroutine
end module
