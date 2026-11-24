program reference_tree_regression_criteria_driver
  use sklearn_tree, only: decision_tree_regressor,extra_tree_regressor
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  class(decision_tree_regressor), allocatable :: model
  integer :: cases,c,mode,n,p,m,criterion,depth,leaf,split,nf,seed,nw,i
  integer(int64) :: decrease_bits
  integer(int64), allocatable :: bits(:)
  real(dp), allocatable :: x(:,:),query(:,:),y(:),weights(:),prediction(:),values(:),impurities(:),mass(:),threshold(:)
  integer, allocatable :: left(:),right(:),features(:),count_(:)
  character(16), parameter :: criteria(5)=[character(16)::'squared_error','friedman_mse','poisson','unknown','absolute_error']
  read(*,*) cases
  do c=1,cases
    read(*,*) mode,n,p,m,criterion,depth,leaf,split,nf,seed,nw,decrease_bits
    allocate(x(n,p),query(m,p),y(n),bits(n*p));read(*,*) bits;x=reshape(transfer(bits,[0._dp],size(bits)),[n,p])
    deallocate(bits);allocate(bits(n));read(*,*) bits;y=transfer(bits,y)
    deallocate(bits);allocate(bits(m*p));read(*,*) bits;query=reshape(transfer(bits,[0._dp],size(bits)),[m,p]);deallocate(bits)
    if(nw>=0) then
      allocate(weights(nw),bits(nw));read(*,*) bits;weights=transfer(bits,weights);deallocate(bits)
    end if
    if(mode==0) then
      allocate(decision_tree_regressor::model)
    else
      allocate(extra_tree_regressor::model)
    end if
    model%criterion=criteria(criterion+1);model%max_depth=depth;model%min_samples_leaf=leaf
    model%min_samples_split=split;model%max_features=nf;model%random_state=seed
    model%min_impurity_decrease=transfer(decrease_bits,0._dp)
    write(*,'(A,1X,I0)') 'BEGIN',c
    call model%fit(x,y,weights)
    prediction=model%predict(query);call emit_r('prediction',prediction)
    call emit_i('leaves',model%apply(query));call emit_r('importance',model%feature_importances_)
    call emit_i('metadata',[model%node_count_,model%get_depth(),model%get_n_leaves(),model%n_features_in_,model%max_features_])
    allocate(values(model%node_count_),impurities(model%node_count_),mass(model%node_count_),threshold(model%node_count_))
    allocate(left(model%node_count_),right(model%node_count_),features(model%node_count_),count_(model%node_count_))
    do i=1,model%node_count_
      associate(node=>model%nodes_(i))
        values(i)=node%value;impurities(i)=node%impurity;mass(i)=node%weighted_n_samples;threshold(i)=node%threshold
        left(i)=node%left;right(i)=node%right;features(i)=node%feature;count_(i)=node%n_samples
      end associate
    end do
    call emit_r('values',values);call emit_r('impurity',impurities);call emit_r('mass',mass)
    call emit_r('threshold',threshold);call emit_i('left',left);call emit_i('right',right)
    call emit_i('feature',features);call emit_i('count',count_)
    call emit_i('criterion_preserved',[merge(1,0,model%criterion==criteria(criterion+1))])
    deallocate(x,query,y,model,values,impurities,mass,threshold,left,right,features,count_)
    if(allocated(weights)) deallocate(weights)
    write(*,'(A)') 'END'
  end do
contains
  subroutine emit_i(name,a)
    character(*), intent(in) :: name
    integer, intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,I0))') name,size(a),a
  end subroutine
  subroutine emit_r(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,ES26.17E3))') name,size(a),a
  end subroutine
end program
